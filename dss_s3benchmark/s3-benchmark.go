// s3-benchmark.go
// Copyright (c) 2017 Wasabi Technology, Inc.
package main
/*
#include "rdd_cl.h"
#include "dss.h"
#include <stdio.h>
#include <stdlib.h>
struct rdd_client_ctx_s *g_rdd_cl_ctx = NULL;
rdd_cl_conn_ctx_t *one_rdd_conn = NULL;
DSSClient dss_client = NULL;

int create_rdd_connection(void* rdd_ip, void* rdd_port, uint16_t* rQhandle) {
  rdd_cl_ctx_params_t param = {RDD_PD_GLOBAL};
  g_rdd_cl_ctx = rdd_cl_init(param);

  rdd_cl_conn_params_t rdd_params;
  rdd_params.ip = (char*)rdd_ip;
  rdd_params.port = (char*)rdd_port;
  printf("About to open S3Bench rdd connection to ip = %s, port = %s \n", rdd_params.ip, rdd_params.port);
  one_rdd_conn = rdd_cl_create_conn(g_rdd_cl_ctx, rdd_params);
  *rQhandle = one_rdd_conn->qhandle;
  return 0;
}

int get_rkey(void* buff, uint64_t buff_len, uint32_t* rkey) {

  struct ibv_mr *mr = rdd_cl_conn_get_mr(one_rdd_conn, buff, buff_len);
  *rkey = mr->rkey;
  return 0;
}

int DSSClient_Init(void *ip_port, void* user, void* passwd, void* uuid, int endpoints_per_cluster) {
  
  dss_client = DSSClientInit((char*)ip_port, (char*)user, (char*)passwd, (char*)uuid, endpoints_per_cluster);
  if (!dss_client) {
    printf("DSS Client init failed for ip_port = %s, uuid = %s, minio_user = %s, minio_pwd = %", ip_port, uuid, user, passwd);
    return -1;
  }
  return 0;
}

int DSSPutObjectBuffer(void* obj_name, int key_length, void* buffer, long int content_length) {
  if (!dss_client) {
    printf("Bailing out PutObjectBuffer as dss_client is NULL");
    return -1;
  }
  //printf("In DSSPutObjectBuffer, key = %s, content_length = %ld\n", (char*)obj_name, content_length);
  if (PutObjectBuffer(dss_client, obj_name, key_length, buffer, content_length) != 0) {
    printf("PutObjectBuffer failed, key = %s, content_length = %l", obj_name, content_length);
    return -1;
  }
  return 0;
}

int DSSGetObjectBuffer(void* obj_name, int key_length, void* buffer, long int content_length, uint32_t* actualLength) {
  if (!dss_client) {
    printf("Bailing out GetObjectBuffer as dss_client is NULL");
    return -1;
  }
  *actualLength = GetObjectBuffer(dss_client, obj_name, key_length, buffer, content_length);
  if (*actualLength == -1) {
    printf("GetObjectBuffer failed, key = %s, content_length = %l", obj_name, content_length);
    return -1;
  }
  return 0;
}


*/
import "C"

import (
	"bytes"
	"crypto/hmac"
	"crypto/md5"
	"crypto/sha1"
	"crypto/tls"
	"encoding/base64"
	"flag"
	"fmt"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	//"github.com/pivotal-golang/bytefmt"
        "code.cloudfoundry.org/bytefmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"net"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
        "unsafe" 
)

// Global variables
var access_key, secret_key, url_host, bucket, region, key_prefix, rdd_ips, rdd_port, dss_ip_port, dss_user, dss_pwd string
var duration_secs, threads, loops, num_ios, num_total_object_per_thread, op_type, is_rdd_get, instance_id, using_client_lib, dss_ep_cluster int
var object_size uint64
var object_data []byte
var object_data_md5 string
var running_threads, upload_count, download_count, delete_count, upload_slowdown_count, download_slowdown_count, delete_slowdown_count int32
var endtime, upload_finish, download_finish, delete_finish time.Time

var s3benchMaxValueSize = getS3BenchMaxValueSize()

func getS3BenchMaxValueSize() int {
        str := os.Getenv("S3BENCH_MAX_VALUE_SIZE")
        if str == "" {
                return 1 * 1024 * 1024
        }
        valSize, err := strconv.Atoi(str)
        if (err != nil) {
          fmt.Println("ERROR:: parsing S3BENCH_MAX_VALUE_SIZE = ", str)
        }
        return valSize
}


var S3ValuePoolData = sync.Pool{
        New: func() interface{} {
                b := make([]byte, s3benchMaxValueSize)
                return &b
        },
}


func logit(msg string) {
	fmt.Println(msg)
	logfile, _ := os.OpenFile("benchmark.log", os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	if logfile != nil {
		logfile.WriteString(time.Now().Format(http.TimeFormat) + ": " + msg + "\n")
		logfile.Close()
	}
}

// Our HTTP transport used for the roundtripper below
var HTTPTransport http.RoundTripper = &http.Transport{
	Proxy: http.ProxyFromEnvironment,
	Dial: (&net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	}).Dial,
	TLSHandshakeTimeout:   10 * time.Second,
	ExpectContinueTimeout: 0,
	// Allow an unlimited number of idle connections
	MaxIdleConnsPerHost: 4096,
	MaxIdleConns:        0,
	// But limit their idle time
	IdleConnTimeout: time.Minute,
	// Ignore TLS errors
	TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
}

var httpClient = &http.Client{Transport: HTTPTransport}

func getS3Client() *s3.S3 {
	// Build our config
	creds := credentials.NewStaticCredentials(access_key, secret_key, "")
	loglevel := aws.LogOff
	// Build the rest of the configuration
	awsConfig := &aws.Config{
		Region:               aws.String(region),
		Endpoint:             aws.String(url_host),
		Credentials:          creds,
		LogLevel:             &loglevel,
		S3ForcePathStyle:     aws.Bool(true),
		S3Disable100Continue: aws.Bool(true),
		// Comment following to use default transport
		HTTPClient: &http.Client{Transport: HTTPTransport},
	}
	session := session.New(awsConfig)
	client := s3.New(session)
	if client == nil {
		log.Fatalf("FATAL: Unable to create new client.")
	}
	// Return success
	return client
}

func createBucket(ignore_errors bool) {
	// Get a client
	client := getS3Client()
	// Create our bucket (may already exist without error)
	in := &s3.CreateBucketInput{Bucket: aws.String(bucket)}
	if _, err := client.CreateBucket(in); err != nil {
		if ignore_errors {
			log.Printf("WARNING: createBucket %s error, ignoring %v", bucket, err)
		} else {
			log.Fatalf("FATAL: Unable to create bucket %s (is your access and secret correct?): %v", bucket, err)
		}
	}
}

func deleteAllObjects() {
	// Get a client
	client := getS3Client()
	// Use multiple routines to do the actual delete
	var doneDeletes sync.WaitGroup
	// Loop deleting our versions reading as big a list as we can
	var keyMarker, versionId *string
	var err error
	for loop := 1; ; loop++ {
		// Delete all the existing objects and versions in the bucket
		in := &s3.ListObjectVersionsInput{Bucket: aws.String(bucket), KeyMarker: keyMarker, VersionIdMarker: versionId, MaxKeys: aws.Int64(1000)}
		if listVersions, listErr := client.ListObjectVersions(in); listErr == nil {
			delete := &s3.Delete{Quiet: aws.Bool(true)}
			for _, version := range listVersions.Versions {
				delete.Objects = append(delete.Objects, &s3.ObjectIdentifier{Key: version.Key, VersionId: version.VersionId})
			}
			for _, marker := range listVersions.DeleteMarkers {
				delete.Objects = append(delete.Objects, &s3.ObjectIdentifier{Key: marker.Key, VersionId: marker.VersionId})
			}
			if len(delete.Objects) > 0 {
				// Start a delete routine
				doDelete := func(bucket string, delete *s3.Delete) {
					if _, e := client.DeleteObjects(&s3.DeleteObjectsInput{Bucket: aws.String(bucket), Delete: delete}); e != nil {
						err = fmt.Errorf("DeleteObjects unexpected failure: %s", e.Error())
					}
					doneDeletes.Done()
				}
				doneDeletes.Add(1)
				go doDelete(bucket, delete)
			}
			// Advance to next versions
			if listVersions.IsTruncated == nil || !*listVersions.IsTruncated {
				break
			}
			keyMarker = listVersions.NextKeyMarker
			versionId = listVersions.NextVersionIdMarker
		} else {
			// The bucket may not exist, just ignore in that case
			if strings.HasPrefix(listErr.Error(), "NoSuchBucket") {
				return
			}
			err = fmt.Errorf("ListObjectVersions unexpected failure: %v", listErr)
			break
		}
	}
	// Wait for deletes to finish
	doneDeletes.Wait()
	// If error, it is fatal
	if err != nil {
		log.Fatalf("FATAL: Unable to delete objects from bucket: %v", err)
	}
}

// canonicalAmzHeaders -- return the x-amz headers canonicalized
func canonicalAmzHeaders(req *http.Request) string {
	// Parse out all x-amz headers
	var headers []string
	for header := range req.Header {
		norm := strings.ToLower(strings.TrimSpace(header))
		if strings.HasPrefix(norm, "x-amz") {
			headers = append(headers, norm)
		}
	}
	// Put them in sorted order
	sort.Strings(headers)
	// Now add back the values
	for n, header := range headers {
		headers[n] = header + ":" + strings.Replace(req.Header.Get(header), "\n", " ", -1)
	}
	// Finally, put them back together
	if len(headers) > 0 {
		return strings.Join(headers, "\n") + "\n"
	} else {
		return ""
	}
}

func hmacSHA1(key []byte, content string) []byte {
	mac := hmac.New(sha1.New, key)
	mac.Write([]byte(content))
	return mac.Sum(nil)
}

func setSignature(req *http.Request) {
	// Setup default parameters
	dateHdr := time.Now().UTC().Format("20060102T150405Z")
	req.Header.Set("X-Amz-Date", dateHdr)
	// Get the canonical resource and header
	canonicalResource := req.URL.EscapedPath()
	canonicalHeaders := canonicalAmzHeaders(req)
	stringToSign := req.Method + "\n" + req.Header.Get("Content-MD5") + "\n" + req.Header.Get("Content-Type") + "\n\n" +
		canonicalHeaders + canonicalResource
	hash := hmacSHA1([]byte(secret_key), stringToSign)
	signature := base64.StdEncoding.EncodeToString(hash)
	req.Header.Set("Authorization", fmt.Sprintf("AWS %s:%s", access_key, signature))
}

func runUpload(thread_num int) {
        var bufp *[]byte = nil
        if (num_ios == 0) {
	  for time.Now().Before(endtime) {
		objnum := atomic.AddInt32(&upload_count, 1)
                if (using_client_lib != 0) {
                  prefix := fmt.Sprintf("%s-Object-%d-%d\000", key_prefix, thread_num, objnum)
                  key := []byte(prefix)
                  if (bufp == nil) { 
                    bufp = S3ValuePoolData.Get().(*[]byte)
                    defer S3ValuePoolData.Put(bufp)
                  }
                  //buf := *bufp
                  //rand.Read(object_data)
                  //if (C.DSSPutObjectBuffer(unsafe.Pointer(&key[0]), unsafe.Pointer(&object_data[0]), C.long(len(object_data))) != 0) {
                  //if (C.DSSPutObjectBuffer(unsafe.Pointer(&key[0]), unsafe.Pointer(&buf[0]), C.long(len(buf))) != 0) {
                  if (C.DSSPutObjectBuffer(unsafe.Pointer(&key[0]), C.int(len(key)), unsafe.Pointer(&(*bufp)[0]), C.long(len(*bufp))) != 0) {
                    log.Fatalf("FATAL: Error uploading object %s:", prefix)
                  }
                } else {

		  fileobj := bytes.NewReader(object_data)
		  prefix := fmt.Sprintf("%s/%s/%s-Object-%d-%d", url_host, bucket, key_prefix, thread_num, objnum)
		  req, _ := http.NewRequest("PUT", prefix, fileobj)
		  req.Header.Set("Content-Length", strconv.FormatUint(object_size, 10))
		  req.Header.Set("Content-MD5", object_data_md5)
		  setSignature(req)
		  if resp, err := httpClient.Do(req); err != nil {
			log.Fatalf("FATAL: Error uploading object %s: %v", prefix, err)
		  } else if resp != nil && resp.StatusCode != http.StatusOK {
                        fmt.Printf("Upload status %s: resp: %+v\n", resp.Status, resp)
			if (resp.StatusCode == http.StatusServiceUnavailable) {
				atomic.AddInt32(&upload_slowdown_count, 1)
				atomic.AddInt32(&upload_count, -1)
			} else {
				fmt.Printf("Upload status %s: resp: %+v\n", resp.Status, resp)
				if resp.Body != nil {
					body, _ := ioutil.ReadAll(resp.Body)
					fmt.Printf("Body: %s\n", string(body))
				}
			}
		  }
                }
	  }
        } else {
          for iter := 0; iter < num_ios; iter++  {
                atomic.AddInt32(&upload_count, 1)
                if (using_client_lib != 0) {
                  prefix := fmt.Sprintf("%s-Object-%d-%d\000", key_prefix, thread_num, iter)
                  key := []byte(prefix)
                  if (bufp == nil) {
                    bufp = S3ValuePoolData.Get().(*[]byte)
                    defer S3ValuePoolData.Put(bufp)
                  }
                  //buf := *bufp

                  //if (C.DSSPutObjectBuffer(unsafe.Pointer(&key[0]), unsafe.Pointer(&object_data[0]), C.long(len(object_data))) != 0) {
                  //if (C.DSSPutObjectBuffer(unsafe.Pointer(&key[0]), unsafe.Pointer(&buf[0]), C.long(len(buf))) != 0) {
                  if (C.DSSPutObjectBuffer(unsafe.Pointer(&key[0]), C.int(len(key)), unsafe.Pointer(&(*bufp)[0]), C.long(len(*bufp))) != 0) {
                    log.Fatalf("FATAL: Error uploading object %s:", prefix)
                  }
                } else {

                  fileobj := bytes.NewReader(object_data)
                  prefix := fmt.Sprintf("%s/%s/%s-Object-%d-%d", url_host, bucket, key_prefix, thread_num, iter)
                  req, _ := http.NewRequest("PUT", prefix, fileobj)
                  req.Header.Set("Content-Length", strconv.FormatUint(object_size, 10))
                  req.Header.Set("Content-MD5", object_data_md5)
                  setSignature(req)
                  if resp, err := httpClient.Do(req); err != nil {
                        log.Fatalf("FATAL: Error uploading object %s: %v", prefix, err)
                  } else if resp != nil && resp.StatusCode != http.StatusOK {
                        fmt.Printf("Upload status %s: resp: %+v\n", resp.Status, resp)
                        if (resp.StatusCode == http.StatusServiceUnavailable) {
                                atomic.AddInt32(&upload_slowdown_count, 1)
                                atomic.AddInt32(&upload_count, -1)
                        } else {
                                fmt.Printf("Upload status %s: resp: %+v\n", resp.Status, resp)
                                if resp.Body != nil {
                                        body, _ := ioutil.ReadAll(resp.Body)
                                        fmt.Printf("Body: %s\n", string(body))
                                }
                        }
                  }
                }  
          }

        }  
	// Remember last done time
	upload_finish = time.Now()
	// One less thread
	atomic.AddInt32(&running_threads, -1)
}

func runDownload(thread_num int) {
   
        var bufp *[]byte = nil
        var rKey C.uint = 0
        //var qHandle C.ushort = 0
        var rddParam string = ""

        if (is_rdd_get != 0) { 
          bufp = S3ValuePoolData.Get().(*[]byte)
          defer S3ValuePoolData.Put(bufp)
          buf := *bufp
          C.get_rkey(unsafe.Pointer(&buf[0]), C.ulong(len(buf)), &rKey)
          //fmt.Println("Got addr, len, rkey = ", &buf[0], len(buf), rKey)
          rddParam = fmt.Sprintf("%x-rdd-%d-rdd-%x-rdd-%d-rdd-", &buf[0], len(buf), rKey, instance_id)
          
        }
        
        
        if (num_ios == 0) {
	  for time.Now().Before(endtime) {
		//atomic.AddInt32(&download_count, 1)
                var objnum int32
                if (num_total_object_per_thread == 0) {
		  objnum = rand.Int31n(upload_count)
                } else {
                  objnum = rand.Int31n(int32 (num_total_object_per_thread))
                }
                if ((objnum == 0) && (num_total_object_per_thread == 0)) {
                        objnum = 1 
                }
                prefix := ""
                if (using_client_lib != 0) {
                  if (bufp == nil) {
                    bufp = S3ValuePoolData.Get().(*[]byte)
                    defer S3ValuePoolData.Put(bufp)
                  }
                  if (num_total_object_per_thread == 0) {
                    prefix = fmt.Sprintf("%s%s-Object-%d\000", rddParam, key_prefix, objnum)
                  } else {
                    prefix = fmt.Sprintf("%s%s-Object-%d-%d\000", rddParam, key_prefix, thread_num, objnum)
                  }
                  key := []byte(prefix)
                  //buf := *bufp
                  var actualLen C.uint = 0
                  //if (C.DSSGetObjectBuffer(unsafe.Pointer(&key[0]), unsafe.Pointer(&buf[0]), C.long(len(buf)), &actualLen) != 0) {
                  if (C.DSSGetObjectBuffer(unsafe.Pointer(&key[0]), C.int(len(key)), unsafe.Pointer(&(*bufp)[0]), C.long(len(*bufp)), &actualLen) != 0) {
                    log.Fatalf("FATAL: Error downloading object %s:", prefix)
                  }       
                } else {
                  if (num_total_object_per_thread == 0) {
		    prefix = fmt.Sprintf("%s/%s/%s%s-Object-%d", url_host, bucket, rddParam, key_prefix, objnum)
                  } else {
                    prefix = fmt.Sprintf("%s/%s/%s%s-Object-%d-%d", url_host, bucket, rddParam, key_prefix, thread_num, objnum)
                  }
                  //fmt.Println("##### GET Req = ", prefix)
		  req, _ := http.NewRequest("GET", prefix, nil)
		  setSignature(req)
		  if resp, err := httpClient.Do(req); err != nil {
			log.Fatalf("FATAL: Error downloading object %s: %v", prefix, err)
		  } else if resp != nil && resp.Body != nil {
                        if (resp.StatusCode != http.StatusOK) {
                          fmt.Printf("Download status %s: resp: %+v\n", resp.Status, resp)
                        }
			if (resp.StatusCode == http.StatusServiceUnavailable){
				atomic.AddInt32(&download_slowdown_count, 1)
				atomic.AddInt32(&download_count, -1)
			} else {
				io.Copy(ioutil.Discard, resp.Body)
			}
		  }
                }  
                atomic.AddInt32(&download_count, 1)
          }
	} else {
          for iter := 0; iter < num_ios; iter++ {
                
                //atomic.AddInt32(&download_count, 1)
                objnum := rand.Int31n(int32(num_ios))
                /*if objnum == 0 {
                        objnum = 1
                }*/
                if (using_client_lib != 0) {
                  if (bufp == nil) {
                    //fmt.Println("Pool alloc..")
                    bufp = S3ValuePoolData.Get().(*[]byte)
                    defer S3ValuePoolData.Put(bufp)
                  }
                  prefix := fmt.Sprintf("%s%s-Object-%d-%d\000", rddParam, key_prefix, thread_num, objnum)
                  key := []byte(prefix)
                  //buf := *bufp
                  var actualLen C.uint = 0
                  //fmt.Println("##### DSS Client GET Req = ", prefix, len(buf))
                  //fmt.Println("Press the Enter Key to stop anytime")
                  //fmt.Scanln()
                  //if (C.DSSGetObjectBuffer(unsafe.Pointer(&key[0]), unsafe.Pointer(&buf[0]), C.long(len(buf)), &actualLen) != 0) {
                  if (C.DSSGetObjectBuffer(unsafe.Pointer(&key[0]), C.int(len(key)), unsafe.Pointer(&(*bufp)[0]), C.long(len(*bufp)), &actualLen) != 0) {
                    log.Fatalf("FATAL: Error downloading object %s:", prefix)
                  }
                  //fmt.Println("##### DSS Client GET Req Success = ", prefix, len(buf), actualLen)
                } else {

                  prefix := fmt.Sprintf("%s/%s/%s%s-Object-%d-%d", url_host, bucket, rddParam, key_prefix, thread_num, objnum)
                  //fmt.Println("##### GET Req = ", prefix)
                  req, _ := http.NewRequest("GET", prefix, nil)
                  setSignature(req)
                  if resp, err := httpClient.Do(req); err != nil {
                        log.Fatalf("FATAL: Error downloading object %s: %v", prefix, err)
                  } else if resp != nil && resp.Body != nil {
                        if (resp.StatusCode != http.StatusOK) {
                          fmt.Printf("Download status %s: resp: %+v\n", resp.Status, resp)
                        }
                        if (resp.StatusCode == http.StatusServiceUnavailable){
                                atomic.AddInt32(&download_slowdown_count, 1)
                                atomic.AddInt32(&download_count, -1)
                        } else {
                                io.Copy(ioutil.Discard, resp.Body)
                        }
                  }
                }
                atomic.AddInt32(&download_count, 1)
          }
        }  
	// Remember last done time
	download_finish = time.Now()
	// One less thread
	atomic.AddInt32(&running_threads, -1)
}

func runDelete(thread_num int) {

        if (num_ios == 0) {
	  for {
		objnum := atomic.AddInt32(&delete_count, 1)
		if objnum > upload_count {
			break
		}
		prefix := fmt.Sprintf("%s/%s/%s-Object-%d", url_host, bucket, key_prefix, objnum)
		req, _ := http.NewRequest("DELETE", prefix, nil)
		setSignature(req)
		if resp, err := httpClient.Do(req); err != nil {
			log.Fatalf("FATAL: Error deleting object %s: %v", prefix, err)
		} else if (resp != nil && resp.StatusCode == http.StatusServiceUnavailable) {
			atomic.AddInt32(&delete_slowdown_count, 1)
			atomic.AddInt32(&delete_count, -1)
		}
	  }
        } else {

          for iter := 0; iter < num_ios; iter++ {
                atomic.AddInt32(&delete_count, 1)
                prefix := fmt.Sprintf("%s/%s/%s-Object-%d-%d", url_host, bucket, key_prefix, thread_num, iter)
                req, _ := http.NewRequest("DELETE", prefix, nil)
                setSignature(req)
                if resp, err := httpClient.Do(req); err != nil {
                        log.Fatalf("FATAL: Error deleting object %s: %v", prefix, err)
                } else if (resp != nil && resp.StatusCode == http.StatusServiceUnavailable) {
                        atomic.AddInt32(&delete_slowdown_count, 1)
                        atomic.AddInt32(&delete_count, -1)
                }
          }

        }
	// Remember last done time
	delete_finish = time.Now()
	// One less thread
	atomic.AddInt32(&running_threads, -1)
}

func main() {
	// Hello
	fmt.Println("Wasabi benchmark program v2.0")

	// Parse command line
	myflag := flag.NewFlagSet("myflag", flag.ExitOnError)
	myflag.StringVar(&access_key, "a", "", "Access key")
	myflag.StringVar(&secret_key, "s", "", "Secret key")
	myflag.StringVar(&url_host, "u", "http://s3.wasabisys.com", "URL for host with method prefix")
	myflag.StringVar(&bucket, "b", "wasabi-benchmark-bucket", "Bucket for testing")
	myflag.StringVar(&region, "r", "us-east-1", "Region for testing")
	myflag.StringVar(&key_prefix, "p", "s3-bench-minio", "Key prefix to be added during key generation")
	myflag.IntVar(&duration_secs, "d", 60, "Duration of each test in seconds")
	myflag.IntVar(&threads, "t", 1, "Number of threads to run")
	myflag.IntVar(&num_ios, "n", 0, "Number of IOS per thread to run")
	myflag.IntVar(&num_total_object_per_thread, "c", 0, "Number of object per thread written earlier")
	myflag.IntVar(&op_type, "o", 0, "Type of op, 1 = put, 2 = get, 3 = del")
	myflag.IntVar(&loops, "l", 1, "Number of times to repeat test")
	myflag.IntVar(&is_rdd_get, "rdd", 0, "Enable RDMA data direct")
	myflag.IntVar(&using_client_lib, "dss_lib", 0, "Enable IO via dss_client_lib")
	myflag.IntVar(&instance_id, "instance_id", 0, "unique id of the instance")
	myflag.IntVar(&dss_ep_cluster, "ep_count", 2, "dss_client end_point per cluster")
        myflag.StringVar(&rdd_ips, "rdd_ips", "127.0.0.1", "comma separated list of RDD ips to connect")
        myflag.StringVar(&rdd_port, "rdd_port", "1234", "RDD port to connect")
        myflag.StringVar(&dss_ip_port, "dss_ipport", "127.0.0.1", "DSS ip-port to connect")
        myflag.StringVar(&dss_user, "dss_user", "minio", "DSS user to authenticate")
        myflag.StringVar(&dss_pwd, "dss_pwd", "minio123", "DSS pwd to authenticate")

	var sizeArg string
	myflag.StringVar(&sizeArg, "z", "1M", "Size of objects in bytes with postfix K, M, and G")
	if err := myflag.Parse(os.Args[1:]); err != nil {
		os.Exit(1)
	}

	// Check the arguments
	if access_key == "" {
		log.Fatal("Missing argument -a for access key.")
	}
	if secret_key == "" {
		log.Fatal("Missing argument -s for secret key.")
	}
        if ((op_type < 0 ) || (op_type > 3)) {
                log.Fatal("Wrong value for -o for op type, should be 1 = put, 2 = get, 3 = del.")
        }
        if ((op_type == 2) && (num_ios == 0)) {
          if (num_total_object_per_thread == 0) {
            log.Fatal("Wrong parameter, need -c in case of get operation and -n is not supplied")        
          }
        }
	var err error
	if object_size, err = bytefmt.ToBytes(sizeArg); err != nil {
		log.Fatalf("Invalid -z argument for object size: %v", err)
	}

	// Echo the parameters
	logit(fmt.Sprintf("Parameters: url=%s, bucket=%s, region=%s, duration=%d, threads=%d, num_ios=%d, op_type=%d, loops=%d, size=%s, prefix = %s, rdd=%d, rdd_ips=%s, rdd_port=%s, instance_id=%d", url_host, bucket, region, duration_secs, threads, num_ios, op_type, loops, sizeArg, key_prefix, is_rdd_get, rdd_ips, rdd_port, instance_id))

        // Create the bucket and delete all the objects
        if (using_client_lib == 0) {
          createBucket(true)
        }
        //deleteAllObjects()


        if (is_rdd_get != 0 && using_client_lib == 0) {
          rdd_conn_data := string(make([]byte, 1024))
          //uuid, _ := ioutil.ReadFile("/proc/sys/kernel/random/uuid")
          
          ip_list := strings.Split(rdd_ips, ",")
          for i := 0; i < len(ip_list); i++ {
            ip := []byte(ip_list[i])
            port := []byte(rdd_port)
            var qHandle C.ushort = 0
            C.create_rdd_connection(unsafe.Pointer(&ip[0]), unsafe.Pointer(&port[0]), &qHandle)
            fmt.Println("RDD connection is successful", ip_list[i], port, qHandle)
            rdd_conn_data += ip_list[i] + "-" + rdd_port + "::" + strconv.Itoa(int(qHandle))
            if (i != (len(ip_list) -1)) {
              rdd_conn_data += "##"
            } else {
              rdd_conn_data += "##END##"
            }
          }
          //rdd_conn_key := string(uuid) + ".dss.rdd.init"
          rdd_conn_key := strconv.Itoa(instance_id) + ".dss.rdd.init"
          fmt.Println("Key, value = ", rdd_conn_key, rdd_conn_data)
          fileobj := bytes.NewReader([]byte(rdd_conn_data))
          prefix := fmt.Sprintf("%s/%s/%s", url_host, bucket, rdd_conn_key)
          req, _ := http.NewRequest("PUT", prefix, fileobj)
          req.Header.Set("Content-Length", strconv.FormatUint(1024, 10))
          //req.Header.Set("Content-Length", strconv.FormatUint(1024, 10))
          //req.Header.Set("Content-MD5", object_data_md5)
          setSignature(req)
          if resp, err := httpClient.Do(req); err != nil {
            log.Fatalf("FATAL: Error uploading object %s: %v", prefix, err)
          } else if resp != nil && resp.StatusCode != http.StatusOK {
            fmt.Printf("Upload status %s: resp: %+v\n", resp.Status, resp)
            if (resp.StatusCode == http.StatusServiceUnavailable) {

            } else {
              fmt.Printf("Upload status %s: resp: %+v\n", resp.Status, resp)
              if resp.Body != nil {
                body, _ := ioutil.ReadAll(resp.Body)
                fmt.Printf("Body: %s\n", string(body))
              }
            }
          }
          //os.Exit(0)  
        }

        if (using_client_lib != 0) {
          //uuid_t, _ := ioutil.ReadFile("/proc/sys/kernel/random/uuid")
          //uuid := string(uuid_t) + "ID : " + strconv.Itoa(instance_id)
          uuid := strconv.Itoa(instance_id) + "\000"
          uuid_b := []byte(uuid)
          ip_port := []byte(dss_ip_port)
          dss_user = access_key
          dss_pwd = secret_key
          dss_u := []byte(dss_user)
          dss_p := []byte(dss_pwd)
          fmt.Println("## Creating DSSClient connection with : ", uuid, dss_ip_port, access_key, secret_key)
          if (C.DSSClient_Init(unsafe.Pointer(&ip_port[0]), unsafe.Pointer(&dss_u[0]), unsafe.Pointer(&dss_p[0]), unsafe.Pointer(&uuid_b[0]), C.int(dss_ep_cluster)) != 0) {
            log.Fatalf("DSS client registration failied for ip_port = %s, dss user = %s, dss_key = %s", dss_ip_port, dss_user, dss_pwd)
          }
        }
	// Initialize data for the bucket
	object_data = make([]byte, object_size)
	rand.Read(object_data)
	hasher := md5.New()
	hasher.Write(object_data)
	object_data_md5 = base64.StdEncoding.EncodeToString(hasher.Sum(nil))

	// Create the bucket and delete all the objects
	//createBucket(true)
	//deleteAllObjects()

	// Loop running the tests
	for loop := 1; loop <= loops; loop++ {

		// reset counters
		upload_count = 0
		upload_slowdown_count = 0
		download_count = 0
		download_slowdown_count = 0
		delete_count = 0
		delete_slowdown_count = 0

                if ((op_type == 0) || (op_type == 1)) {
		  // Run the upload case
		  running_threads = int32(threads)
		  starttime := time.Now()
		  endtime = starttime.Add(time.Second * time.Duration(duration_secs))
		  for n := 1; n <= threads; n++ {
			go runUpload(n)
		  }

		  // Wait for it to finish
		  for atomic.LoadInt32(&running_threads) > 0 {
			time.Sleep(time.Millisecond)
		  }
		  upload_time := upload_finish.Sub(starttime).Seconds()

		  bps := float64(uint64(upload_count)*object_size) / upload_time
		  logit(fmt.Sprintf("Loop %d: PUT time %.1f secs, objects = %d, speed = %sB/sec, %.1f operations/sec. Slowdowns = %d",
		        loop, upload_time, upload_count, bytefmt.ByteSize(uint64(bps)), float64(upload_count)/upload_time, upload_slowdown_count))
                }

                if ((op_type == 0) || (op_type == 2)) {
		  // Run the download case
		  running_threads = int32(threads)
		  starttime := time.Now()
		  endtime = starttime.Add(time.Second * time.Duration(duration_secs))
		  for n := 1; n <= threads; n++ {
			go runDownload(n)
		  }

		  // Wait for it to finish
		  for atomic.LoadInt32(&running_threads) > 0 {
			time.Sleep(time.Millisecond)
		  }
		  download_time := download_finish.Sub(starttime).Seconds()

		  bps := float64(uint64(download_count)*object_size) / download_time
		  logit(fmt.Sprintf("Loop %d: GET time %.1f secs, objects = %d, speed = %sB/sec, %.1f operations/sec. Slowdowns = %d",
			loop, download_time, download_count, bytefmt.ByteSize(uint64(bps)), float64(download_count)/download_time, download_slowdown_count))
                }

                if ((op_type == 0) || (op_type == 3)) {
		  // Run the delete case
		  running_threads = int32(threads)
		  starttime := time.Now()
		  endtime = starttime.Add(time.Second * time.Duration(duration_secs))
		  for n := 1; n <= threads; n++ {
			go runDelete(n)
		  }

		  // Wait for it to finish
		  for atomic.LoadInt32(&running_threads) > 0 {
			time.Sleep(time.Millisecond)
		  }
		  delete_time := delete_finish.Sub(starttime).Seconds()

		  logit(fmt.Sprintf("Loop %d: DELETE time %.1f secs, %.1f deletes/sec. Slowdowns = %d",
			loop, delete_time, float64(delete_count)/delete_time, delete_slowdown_count))
                }
	}

	// All done
}
