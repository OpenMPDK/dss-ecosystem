package main
import (
    "fmt"
    "os"
    "flag"
    "sync/atomic"
    "math/rand"
    "io/ioutil"
    "log"
    "sync"
    "time"
    "strconv"
    "code.cloudfoundry.org/bytefmt"
    "runtime/debug"
    //"github.com/nfnt/resize"
    //"image"
    //"image/png"
)

func help() {
  fmt.Println("-n, type=int, required=True, Desired number of directories to generate")
  fmt.Println("-f, type=int, required=True, Max number of files per directory, randomly generate 1 - N number of files per directory")
  fmt.Println("-t, type=int, required=True, Amount of data in GB to generate")
  fmt.Println("-p, type=str, required=True, Directory name to generate files to")
  fmt.Println("-d, type=int, required=Optional, default=10, Maximum depth of directories to calculate")
  fmt.Println("-w, type=int, required=Optional, default=10, Maximum width of directories fo calculate")
  fmt.Println("-maxSize, type=int, required=Optional, default=1048576, Maximum file size in Bytes")
  fmt.Println("-minSize, type=int, required=Optional, default=524288, Minimum file size in Bytes")

}

var numDir,numFiles,maxDepth,maxWidth,maxFilesPerThread,numExtra,numShard int
var total_dir int32 = 0
var total_files int64 = 0
var totalSize,maxfSize,minfSize int64
var rootPrefix string
var fPrefix string
var modify_image int = 0

func ceilFrac(numerator, denominator int) (ceil int, remainder int) {
        if denominator == 0 {
                // do nothing on invalid input
                return
        }
        // Make denominator positive
        if denominator < 0 {
                numerator = -numerator
                denominator = -denominator
        }
        ceil = numerator / denominator
        remainder = numerator%denominator
        if numerator > 0 && remainder != 0 {
                ceil++
        }
        return
}


var buffPool = sync.Pool{
        New: func() interface{} {
                b := make([]byte, maxfSize)
                return &b
        },
}


func create_img_files (prefix_str string, numFilesInthis int, totalWSize* int64) {

          fileSizeInthis := rand.Int63n(maxfSize - minfSize + 1) + minfSize
          //fileSizeInthis := maxfSize
          var wg = &sync.WaitGroup{}

          for iFile := 0; iFile < numFilesInthis; iFile++ {
            if (atomic.LoadInt64(totalWSize) <= 0) {
              //fmt.Println("create_files returning as total length achieved..", prefix_str, numFilesInthis)
              return
            }
            wg.Add(1)
            go func(prefix_str string, iFile int, fileSizeInthis int64) {
                defer wg.Done()
                bufp := buffPool.Get().(*[]byte)
                defer buffPool.Put(bufp)
                //object_data := make([]byte, fileSizeInthis)
                object_data := (*bufp)[:fileSizeInthis]
                rand.Read(object_data[:256])
                f_name := prefix_str + "/" + fPrefix + "_" + strconv.Itoa(iFile)
                if (atomic.LoadInt64(totalWSize) <= 0) {
                  //fmt.Println("create_files returning as total length achieved..", prefix_str, numFilesInthis)
                  return
                }

                err := ioutil.WriteFile(f_name, object_data, 0644)
                if err != nil {
                  log.Fatal(err)
                }
                atomic.AddInt64(&total_files, 1)
                atomic.AddInt64(totalWSize, ^int64(fileSizeInthis-1))
                //fmt.Println("create_files file created = ", f_name, fileSizeInthis, *totalWSize, numFilesInthis )
            }(prefix_str, iFile, fileSizeInthis)

          }
          wg.Wait()


}

func create_img_files_b (prefix_str string, numFilesInthis int, totalSizeInthis int64, img_buf []byte, wg_main *sync.WaitGroup) {
//func create_img_files_b (prefix_str string, numFilesInthis int, totalSizeInthis int64, img_buf image.Image, wg_main *sync.WaitGroup) {
 
          defer wg_main.Done()
          numBatch,numFilesLastBatch := ceilFrac(numFilesInthis, maxFilesPerThread)
          //fileSizeInthis := rand.Int63n(maxfSize - minfSize + 1) + minfSize
          //fileSizeInthis := maxfSize
          fileSizeInthis := int64(len(img_buf))
          var wg = &sync.WaitGroup{}
          //fmt.Println("create_img_files_b :: ", numBatch, numFilesLastBatch, fileSizeInthis, maxFilesPerThread)
          for iBatch := 0; iBatch < numBatch; iBatch++ {
            if (atomic.LoadInt64(&totalSizeInthis) <= 0) {
              fmt.Println("create_files returning as total length achieved..", prefix_str, numFilesInthis)
              return
            }
            numFilesTowrite := maxFilesPerThread
            if (iBatch + 1 == numBatch && numFilesLastBatch != 0) {
              numFilesTowrite = numFilesLastBatch
            }
            wg.Add(1)
            go func(prefix_str string, numFilesTowrite int, iBatch int, fileSizeInthis int64) {
              defer wg.Done()
              //bufp := buffPool.Get().(*[]byte)
              //defer buffPool.Put(bufp)
              var object_data []byte = nil
              if (modify_image != 0) {
                object_data = make([]byte, fileSizeInthis)
              }
              for iFile := 0; iFile < numFilesTowrite; iFile++ {
                //object_data := make([]byte, fileSizeInthis)
                //object_data := (*bufp)[:fileSizeInthis]
                if (modify_image != 0) {
                  copy(object_data, img_buf)
                  rand.Read(object_data[:256])
                }
                f_name := prefix_str + "/" + fPrefix + "_" + strconv.Itoa(iBatch) + "_" + strconv.Itoa(iFile) + ".png"
                //fmt.Println("create_nested_dir_file file creating = ", f_name, fileSizeInthis, totalSizeInthis, numFilesInthis, numFilesTowrite)

                if (modify_image != 0) {
                  err := ioutil.WriteFile(f_name, object_data, 0644)
              
                  if err != nil {
                    log.Fatal(err)
                  }
                } else {
                  err := ioutil.WriteFile(f_name, img_buf, 0644)

                  if err != nil {
                    log.Fatal(err)
                  }

                }
                /*img_out_f,_ := os.Create(f_name)
                png.Encode(img_out_f, img_buf)
                img_out_f.Close()*/
                atomic.AddInt64(&total_files, 1)
                atomic.AddInt64(&totalSizeInthis, ^int64(fileSizeInthis-1))
                //time.Sleep(10 * time.Millisecond)
              }
            }(prefix_str, numFilesTowrite, iBatch, fileSizeInthis)
            time.Sleep(1 * time.Millisecond)
          }
          wg.Wait()

}


func main() {

    /*argsWithProg := os.Args
    argsWithoutProg := os.Args[1:]
    arg := os.Args[3]*/
    debug.SetMaxThreads(100000)
    numFilesptr := flag.Int("f", 100, "Max number of files per directory, randomly generate 0 - N number of files per directory")
    totalSizeptr := flag.Int64("size", 100, "Amount of data in MB to generate")
    maxfSizeptr := flag.Int64("maxSize", 1048576, "Maximum file size in Bytes")
    minfSizeptr := flag.Int64("minSize", 524288, "Minimum file size in Bytes")
    maxFilesPerThreadptr := flag.Int("maxFilesPerThread", 128, "Max number of files per thread to batch")
    maxShardptr := flag.Int("maxShard", 1, "Max number of shard to use")
    seedVal := flag.Int64("seed", 0, "Random seed to use, default time")
    dirPrefix := flag.String("p", "/mnt/nfs_share/", "Directory name to generate files to")
    filePrefix := flag.String("filePrefix", "pytorch_dummy", "File name prefix")
    appPrefix := flag.String("rootPrefix", "root", "root level prefix")
    startImagefile := flag.String("startingFile", "data/sample.png", "Sample img file to copy data from, if not mentioned, it will create dummy buffer")
    modify_image_p := flag.Int("modifyImage", 0, "Modify each image randomly ?")
    flag.Parse()
    numFiles = *numFilesptr
    totalSize = *totalSizeptr*1024*1024
    maxfSize = *maxfSizeptr
    minfSize = *minfSizeptr
    maxFilesPerThread = *maxFilesPerThreadptr
    rootPrefix = *dirPrefix
    fPrefix = *filePrefix
    modify_image = *modify_image_p

    log.Printf("Starting with : numFiles = %d, totalSize = %d, maxfSize = %d, minfSize = %d, seed = %d, num_shard = %d, dir_prefix = %s, file_prefix = %s, top_prefix = %s, img file to copy = %s\n", 
               numFiles, totalSize, maxfSize, minfSize, *seedVal, *maxShardptr, rootPrefix, fPrefix, *appPrefix, *startImagefile)

    sample_img, err_img := ioutil.ReadFile(*startImagefile)
    if err_img != nil {
      log.Fatalf("unable to read file(%s): %v", *startImagefile, err_img)
    }
    maxfSize = int64(len(sample_img))
    /*fi, err_stat := os.Stat(*startImagefile)
    if err_stat != nil {
      log.Fatal(err_stat)
    }
    // get the size
    maxfSize = fi.Size()
    minfSize = maxfSize
    catFile, err := os.Open(*startImagefile)
    if err != nil {
        log.Fatal(err)
    }
    defer catFile.Close()
 
    // Consider using the general image.Decode as it can sniff and decode any registered image format.
    sample_img, err := png.Decode(catFile)
    if err != nil {
        log.Fatal(err)
    }*/
    
    if (*seedVal == 0) {
      rand.Seed(time.Now().UnixNano())
    } else {
      rand.Seed(*seedVal)
    }
    numExtra = 0
    numShard = *maxShardptr
    totalPerShard := totalSize / int64(numShard)
    maxPerShard := int64(numFiles) * maxfSize
    if (totalPerShard >= maxPerShard) {
      numShard = 0
    }
    for totalPerShard > maxPerShard {
      numShard++
      totalPerShard = totalSize / int64(numShard)
    }
     
    remainderLast := totalSize % int64(numShard)
    loopCount := numShard

    if (remainderLast != 0) {
      loopCount++
    }
    log.Printf("### Total number of shard/folder = %d, size per shard/folder = %d, remainder = %d", loopCount, totalPerShard, remainderLast)
    var wg = &sync.WaitGroup{} 
    start := time.Now() 
    for iShard := 0; iShard < loopCount; iShard++ {
      dir_str := rootPrefix + "/" + "shard_" + *appPrefix + "_" + strconv.Itoa(iShard)
      os.MkdirAll(dir_str, 0755)
      wg.Add(1)
      if (remainderLast != 0 && (iShard == (loopCount -1))) {
        log.Printf("Starting shard write with : totalSize = %d, totalFiles = %d, root_prefix = %s\n",
               remainderLast, numFiles, dir_str) 
        create_img_files_b(dir_str, numFiles, remainderLast, sample_img, wg)
      } else {
        log.Printf("Starting shard write with : totalSize = %d, totalFiles = %d, root_prefix = %s\n",
               totalPerShard, numFiles, dir_str)

        create_img_files_b(dir_str, numFiles, totalPerShard, sample_img, wg)
      }

    }
    wg.Wait()
    duration := time.Since(start)
    durSec := int64(duration/time.Second)
    var bps int64 = 0 
    if (durSec != 0) {
      bps = (totalSize / durSec)
    }

    log.Printf("Dataset creation completed successfully, num_files = %d, BW = %sB/sec, dureation = %v",
                total_files, bytefmt.ByteSize(uint64(bps)), duration)
}
