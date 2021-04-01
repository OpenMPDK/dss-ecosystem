/**
 *   BSD LICENSE
 *
 *   Copyright (c) 2021 Samsung Electronics Co., Ltd.
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Samsung Electronics Co., Ltd. nor the names of
 *       its contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <mutex>
#include <sys/stat.h>

#include <unistd.h>

#include <aws/core/Aws.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/utils/HashingUtils.h>

#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/CreateBucketRequest.h>
#include <aws/s3/model/DeleteBucketRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/BucketLocationConstraint.h>
#include <aws/s3/model/CommonPrefix.h>

#include "dss.h"

namespace dss {

using namespace Aws;

DSSInit dss_init;

Endpoint::Endpoint(Aws::Auth::AWSCredentials& cred, const std::string& url, Config& cfg) 
{
	cfg.endpointOverride = url.c_str();
	m_ses = Aws::S3::S3Client(cred, cfg, 
						Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false);
}

Result
Endpoint::HeadBucket(const Aws::String& bucket)
{
	Aws::S3::Model::HeadBucketRequest req;
	req.SetBucket(bucket);

	auto&& out = m_ses.HeadBucket(req);

	if (out.IsSuccess())
		return Result(true);
	else
		return Result(false, out.GetError());
}
	
Result
Endpoint::CreateBucket(const Aws::String& bn)
{
    Aws::S3::Model::CreateBucketRequest request;
    request.SetBucket(bn);

    auto out = m_ses.CreateBucket(request);

	if (out.IsSuccess())
		return Result(true);
	else
		return Result(false, out.GetError());
}

Result
Endpoint::GetObject(const Aws::String& bn, Request* req)
{
    Aws::S3::Model::GetObjectRequest ep_req;
    ep_req.WithBucket(bn).SetKey(Aws::String(req->key.c_str()));

    Aws::S3::Model::GetObjectOutcome out = m_ses.GetObject(ep_req);

    if (out.IsSuccess()) {
        return Result(true, out.GetResultWithOwnership());
    } else {
		return Result(false, out.GetError());
    }
}


Result
Endpoint::GetObject(const Aws::String& bn, const Aws::String& objectName)
{
    Aws::S3::Model::GetObjectRequest req;
    req.WithBucket(bn).SetKey(objectName);

    Aws::S3::Model::GetObjectOutcome out = m_ses.GetObject(req);

    if (out.IsSuccess()) {
        return Result(true, out.GetResultWithOwnership());
    } else {
		return Result(false, out.GetError());
#if 0
		auto err = outcome.GetError();
        if (err.GetErrorType() == Aws::S3::S3Errors::RESOURCE_NOT_FOUND)
        	throw NoSuchResourceError();
		else
			throw GenericError(err.GetMessage().c_str());

        return false;
#endif
    }
}

void PutObjectAsyncDone(const Aws::S3::S3Client* s3Client, 
    const Aws::S3::Model::PutObjectRequest& request, 
    const Aws::S3::Model::PutObjectOutcome& outcome,
    const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context)
{
	const std::shared_ptr<const CallbackCtx> ctx = 
			std::static_pointer_cast<const CallbackCtx>(context);

    if (outcome.IsSuccess()) {
        Callback cb = ctx->getCbFunc();
		Request* req = (Request*)ctx->getCbArgs();

        cb(req->done_arg, req->key, req->key, 0);
    } else {
        std::cout << "Error: PutObjectAsyncDone: " <<
            outcome.GetError().GetMessage() << std::endl;
    }

    //upload_variable.notify_one();
}


Result
Endpoint::PutObjectAsync(const Aws::String& bn, Request* req)
{
    // Create and configure the asynchronous put object request.
    Aws::S3::Model::PutObjectRequest request;
    request.WithBucket(bn).SetKey(Aws::String(req->key.c_str()));
	request.SetBody(req->io_stream);

    // Create and configure the context for the asynchronous put object request.

    std::shared_ptr<Aws::Client::AsyncCallerContext> context =
    		Aws::MakeShared<CallbackCtx>("PutObjectAllocationTag", req->done_func, req);
    context->SetUUID(Aws::String(req->key.c_str()));

    // Make the asynchronous put object call. Queue the request into a 
    // thread executor and call the PutObjectAsyncDone function when the 
    // operation has finished. 
    m_ses.PutObjectAsync(request, PutObjectAsyncDone, context);

    return true;
}

Result
Endpoint::PutObject(const Aws::String& bn, const Aws::String& objectName,
					std::shared_ptr<Aws::IOStream>& input_stream) 
{
    S3::Model::PutObjectRequest request;
    request.WithBucket(bn).SetKey(objectName);
    request.SetBody(input_stream);

    S3::Model::PutObjectOutcome out = m_ses.PutObject(request);

    if (out.IsSuccess()) {
        return Result(true);
    } else {
		return Result(false, out.GetError());
    }
}

Result
Endpoint::PutObject(const Aws::String& bn, Request* req) 
{
    S3::Model::PutObjectRequest ep_req;
    ep_req.WithBucket(bn).SetKey(Aws::String(req->key.c_str()));
    ep_req.SetBody(req->io_stream);

    S3::Model::PutObjectOutcome out = m_ses.PutObject(ep_req);

    if (out.IsSuccess()) {
        return Result(true);
    } else {
		return Result(false, out.GetError());
    }
}

Result
Endpoint::DeleteObject(const Aws::String& bn, const Aws::String& objectName)
{
    Aws::S3::Model::DeleteObjectRequest request;

    request.WithBucket(bn).SetKey(objectName);

    auto out = m_ses.DeleteObject(request);

    if (out.IsSuccess()) {
        return Result(true);
    } else {
		return Result(false, out.GetError());
    }
}

Result
Endpoint::DeleteObject(const Aws::String& bn, Request* req)
{
    Aws::S3::Model::DeleteObjectRequest ep_req;

    ep_req.WithBucket(bn).SetKey(Aws::String(req->key.c_str()));

    auto out = m_ses.DeleteObject(ep_req);

    if (out.IsSuccess()) {
        return Result(true);
    } else {
		return Result(false, out.GetError());
    }
}

Result
Endpoint::ListObjects(const Aws::String& bn, Objects *os)
{
	bool cont = false;
	std::string token;
    S3::Model::ListObjectsV2Outcome out;
    S3::Model::ListObjectsV2Request req;

    req.WithBucket(bn).WithPrefix(os->GetPrefix()).WithDelimiter(os->GetDelim().c_str());
    if (os->PageSizeSet())
    	req.SetMaxKeys(os->GetPageSize());
    if (os->TokenSet())
    	req.SetContinuationToken(os->GetToken().c_str());

    do {
        out = m_ses.ListObjectsV2(req);
        if (out.IsSuccess()) {
			//TODO: std::move()
            Aws::Vector<Aws::S3::Model::Object> objects =
                                            out.GetResult().GetContents();
        	Aws::Vector<Aws::S3::Model::CommonPrefix> cps =
        									out.GetResult().GetCommonPrefixes();

            for (auto o : objects)
                os->GetPage().insert(o.GetKey().c_str());
            for (auto cp : cps)
            	os->GetPage().insert(cp.GetPrefix().c_str());

        } else {
			return Result(false, out.GetError());
        }

        if (out.GetResult().GetIsTruncated()) {
			if (os->PageSizeSet()) {
				os->SetToken(out.GetResult().GetNextContinuationToken());
				cont = false;
			} else {
				cont = true;
			}
		} else {
			os->SetToken("");
			cont = false;
		}

    } while (cont &&
    		 (req.SetContinuationToken(out.GetResult().GetNextContinuationToken().c_str()), true));

    return Result(true);
}

int
ClusterMap::DownloadClusterConf()
{
	Result r = m_client->GetClusterConfig();
	if (!r.IsSuccess()) {
		auto err = r.GetErrorType();
		if (err == Aws::S3::S3Errors::NETWORK_CONNECTION)
			throw NetworkError();
 
		throw DiscoverError("Failed to download conf.json: " + r.GetErrorMsg()); 	
		return -1;
	}

	using json = nlohmann::json;

	try {
    	json conf = json::parse(r.GetIOStream());
		for (auto &c : conf["clusters"]) {
			Cluster* cluster = InsertCluster(c["id"]);
			pr_debug("Adding cluster %u\n", (uint32_t)c["id"]);
			for (auto &ep : c["endpoints"])
				cluster->InsertEndpoint(m_client, ep["ipv4"], ep["port"]);
		}
	} catch (std::exception& e) {
		throw DiscoverError("Parse conf.json error: " + Aws::String(e.what()));
	}

	return 0;
}

int
ClusterMap::VerifyClusterConf()
{
	std::vector<bool> empty;
	empty.resize(m_clusters.size());

	{
		std::lock_guard<std::mutex> lock(dss_init.mutex());
		for (auto c : m_clusters) {
			Result r = c->HeadBucket();
        	if (!r.IsSuccess())
				empty[c->GetID()] = true;
		}

		if (!std::equal(empty.begin() + 1, empty.end(), empty.begin())) {
			uint32_t i = 0;
			for (auto it : empty) {
				pr_err("cluster %u : %s\n", i++, (unsigned)it ? "present" : "missing");
			}

			pr_err("DSS buckets are missing\n");
			return -1;
		}

		if (empty[0]) {
			for (auto c : m_clusters) {
				Result r = c->CreateBucket();
				if (!r.IsSuccess()) {
					pr_err("Failded to create bucket on cluster %u (err=%u)\n",
							c->GetID(), (unsigned)r.GetErrorType());
					return -1;
				}
			}
		}

	}

	return 0;
}

void
ClusterMap::GetCluster(Request* req)
{
	unsigned id = 0, max_w = GetCLWeight(0, req->key.c_str());
	for (unsigned i=1; i<m_clusters.size(); i++) {
		unsigned w = GetCLWeight(i, req->key.c_str());
		 pr_debug("key %s: cluster %u weight %0x\n", req->key.c_str(), i, w);
		if (w > max_w) {
			id = i;
			max_w = w;
		}
	}

	req->key_hash = max_w;
	req->cluster = m_clusters[id];

	pr_debug("key %s: cluster %u weight %0x\n", req->key.c_str(), id, max_w);
}

int
Cluster::InsertEndpoint(Client* c, const std::string& ip, uint32_t port)
{
	Endpoint* ep = new Endpoint(c->GetCredential(), ip + ":" + std::to_string(port), c->GetConfig()); 
	m_endpoints.push_back(ep);

	pr_debug("Insert endpoint %s\n", (ip + ":" + std::to_string(port)).c_str());
	
	return 0;
}

Result
Cluster::HeadBucket()
{
	return m_endpoints[0]->HeadBucket(m_bucket);
}

Result
Cluster::CreateBucket()
{
	return m_endpoints[0]->CreateBucket(m_bucket);
}

Result
Cluster::GetObject(const Aws::String& objectName)
{
	return m_endpoints[0]->GetObject(m_bucket, objectName);
}

Result
Cluster::GetObject(Request* r)
{
	return GetEndpoint(r)->GetObject(m_bucket, r);
}

Result
Cluster::PutObject(const Aws::String& objectName, std::shared_ptr<Aws::IOStream>& input_stream)
{
	return std::move(m_endpoints[0]->PutObject(m_bucket, objectName, input_stream));
}

Result
Cluster::PutObjectAsync(Request* r)
{
	return GetEndpoint(r)->PutObjectAsync(m_bucket, r);
}

Result
Cluster::PutObject(Request* r)
{
	return std::move(GetEndpoint(r)->PutObject(m_bucket, r));
}

Result
Cluster::DeleteObject(const Aws::String& objectName)
{
	return m_endpoints[0]->DeleteObject(m_bucket, objectName);
}

Result
Cluster::DeleteObject(Request* r)
{
	return GetEndpoint(r)->DeleteObject(m_bucket, r);
}

Result
Cluster::ListObjects(Objects *objs)
{
	return m_endpoints[0]->ListObjects(m_bucket, objs);
}

Result
Client::GetClusterConfig()
{
	return m_discover_ep->GetObject(DISCOVER_BUCKET, DISCOVER_CONFIG_KEY);
}

Config
Client::ExtractOptions(const SesOptions& o)
{
	Aws::Client::ClientConfiguration cfg;
	cfg.scheme = strncasecmp(o.scheme.c_str(), "https", std::string("https").length()) ? 
					Aws::Http::Scheme::HTTP :
					Aws::Http::Scheme::HTTPS;
	cfg.verifySSL = false;
	cfg.useDualStack = o.useDualStack;
	cfg.maxConnections = o.maxConnections;
	cfg.httpRequestTimeoutMs = o.httpRequestTimeoutMs;
	cfg.requestTimeoutMs = o.requestTimeoutMs;
	cfg.connectTimeoutMs = o.connectTimeoutMs;
	cfg.enableTcpKeepAlive = o.enableTcpKeepAlive;
	cfg.tcpKeepAliveIntervalMs = o.tcpKeepAliveIntervalMs;

	return cfg;
}

std::unique_ptr<Client>
Client::CreateClient(const std::string& ip,
					 const std::string& user, const std::string& pwd, const SesOptions& options)
{
	std::unique_ptr<Client> client(new Client(ip, user, pwd, options));
	if (client->InitClusterMap() < 0) {
		pr_err("Failed to init cluster map\n");
		return nullptr;
	}

	return client;
}

Client::~Client()
{
	delete m_discover_ep;
	delete m_cluster_map;
}

int
Client::InitClusterMap()
{
	ClusterMap* map = new ClusterMap(this);
	if (map->DownloadClusterConf() < 0)
		return -1;
	if (map->VerifyClusterConf() < 0)
		return -1;

	m_cluster_map = map;
	return 0;
}

Result
Request::Submit(Handler handler)
{
	return ((this->cluster->*handler)(this));
}

int Client::GetObject(const Aws::String& objectName, const Aws::String& dest_fn)
{
	std::unique_ptr<Request> req_guard(new Request(objectName.c_str(), dest_fn.c_str()));

	m_cluster_map->GetCluster(req_guard.get());
	Result r = req_guard->Submit(&Cluster::GetObject);

    if (r.IsSuccess()) {
        std::fstream local_file;
        local_file.open(dest_fn.c_str(), std::ios::out | std::ios::binary);
        
        auto& object_stream = r.GetIOStream();

		//TODO: revist performance
        local_file << object_stream.rdbuf();
        local_file.flush();
        local_file.close();
        return true;
    } else {
        auto err = r.GetErrorType();
        if (err == Aws::S3::S3Errors::RESOURCE_NOT_FOUND)
                throw NoSuchResourceError();
        else
            throw GenericError(r.GetErrorMsg().c_str());

        return false;
    }
}

int Client::PutObjectAsync(const std::string& objectName, const std::string& src_fn,
						   Callback cb, void* cb_arg)
{
	Result r;
    struct stat buffer;
	Request* req = new Request(objectName.c_str(), src_fn.c_str(), cb, cb_arg);

    if (stat(src_fn.c_str(), &buffer) == -1) {
        pr_err("Error: PutObject: File '%s' does not exist.",
        		src_fn.c_str());
        return false;
    }

	req->io_stream = Aws::MakeShared<Aws::FStream>("Client::PutObject",
            			src_fn.c_str(),
            			std::ios_base::in | std::ios_base::binary);

	m_cluster_map->GetCluster(req);
	r = std::move(req->Submit(&Cluster::PutObjectAsync));

    if (r.IsSuccess()) {
        return 0;
    } else {
        auto err = r.GetErrorType();
        if (err == Aws::S3::S3Errors::RESOURCE_NOT_FOUND)
                throw NoSuchResourceError();
        else
            throw GenericError(r.GetErrorMsg().c_str());

        return -1;
    }
}

int Client::PutObject(const Aws::String& objectName, const Aws::String& src_fn, bool async)
{
	Result r;
    struct stat buffer;
	std::unique_ptr<Request> req_guard(new Request(objectName.c_str(), src_fn.c_str()));

    if (stat(src_fn.c_str(), &buffer) == -1) {
        pr_err("Error: PutObject: File '%s' does not exist.",
        		src_fn.c_str());
        return false;
    }

	req_guard->io_stream = Aws::MakeShared<Aws::FStream>("Client::PutObject",
            			src_fn.c_str(),
            			std::ios_base::in | std::ios_base::binary);

	m_cluster_map->GetCluster(req_guard.get());
	if (!async)
		r = std::move(req_guard->Submit(&Cluster::PutObject));
	else
		r = std::move(req_guard->Submit(&Cluster::PutObjectAsync));

    if (r.IsSuccess()) {
        return 0;
    } else {
        auto err = r.GetErrorType();
        if (err == Aws::S3::S3Errors::RESOURCE_NOT_FOUND)
                throw NoSuchResourceError();
        else
            throw GenericError(r.GetErrorMsg().c_str());

        return -1;
    }
}

int Client::DeleteObject(const Aws::String& objectName)
{
	std::unique_ptr<Request> req_guard(new Request(objectName.c_str()));
	m_cluster_map->GetCluster(req_guard.get());
	Result r = req_guard->Submit(&Cluster::DeleteObject);

    if (r.IsSuccess()) {
        return 0;
    } else {
        auto err = r.GetErrorType();
        if (err == Aws::S3::S3Errors::RESOURCE_NOT_FOUND)
                throw NoSuchResourceError();
        else
            throw GenericError(r.GetErrorMsg().c_str());

        return -1;
    }
}

std::set<std::string>&&
Client::ListObjects(const std::string& prefix, const std::string& delimit)
{
	std::unique_ptr<Objects> objs = GetObjects(prefix, delimit, 0);
	const std::vector<Cluster*> clusters = m_cluster_map->GetClusters();

	for (auto c : clusters) {
		Result r = c->ListObjects(objs.get());
		if (!r.IsSuccess()) {
			auto err = r.GetErrorType();
	        if (err == Aws::S3::S3Errors::RESOURCE_NOT_FOUND)
                throw NoSuchResourceError();
        	else
            	throw GenericError(r.GetErrorMsg().c_str());
		}
	}

	return std::move(objs->GetPage());
}

int Objects::GetObjKeys() 
{
	const std::vector<Cluster*> clusters = m_cluster_map->GetClusters();

	m_page.clear();

	if (m_cur_id == -1) {
		m_cur_id = 0;
	} else if ((unsigned long)m_cur_id == clusters.size()) {
		m_cur_id = -1;
		return -1;
	}

	while (1) {
		Result r = clusters[m_cur_id]->ListObjects(this);
		if (!r.IsSuccess()) {
			auto err = r.GetErrorType();
			if (err == Aws::S3::S3Errors::RESOURCE_NOT_FOUND)
				throw NoSuchResourceError();
        	else
            	throw GenericError(r.GetErrorMsg().c_str());
		}

		//pr_debug("cluster %u returns %lu keys\n", m_cur_id, m_page.size());

		if (m_page.size() < GetPageSize()) {
			// TODO: It is unclear whether aws sdk would return
			// # of keys less than pagesize while there are still
			// leftovers in cluster, so we rely on GetIsTruncated()
			// which sets token if true;
			if (!TokenSet()) {
				m_cur_id += 1;
				if ((unsigned long)m_cur_id == clusters.size())
					break;
			}
			continue;
		} else {
			if (!TokenSet())
				m_cur_id += 1;
			break;
		}
	}

    return 0;
}

} // namespace dss

void
test_put_done(void* ptr, std::string key, std::string message, int err)
{
	printf("%s: key %s\n", __func__, key.c_str());
}

void*
do_work(void*) 
{
	std::unique_ptr<dss::Client> client
		= dss::Client::CreateClient("http://127.0.0.1:9001", "minioadmin", "minioadmin");
	if (!client)
		fprintf(stderr, "Failed to create client\n");

	sleep(2);

	return NULL;
}

int main()
{
	const Aws::String object_name = "test_obj";
    const Aws::String fname = "/root/jerry/dss_client/src/dss_client.cpp";

    for (int i=0; i<5; i++) {
    	pthread_t t; 
    	pthread_create(&t, NULL, do_work, NULL);	
    }

	std::unique_ptr<dss::Client> client
		= dss::Client::CreateClient("http://127.0.0.1:9001", "minioadmin", "minioadmin");
	if (!client) {
		fprintf(stderr, "Failed to create client\n");
		return -1;
	}

/*
	for (unsigned i=0; i<2; i++) {
		Aws::String key = object_name + std::to_string(i).c_str();
       	client->PutObject(key, fname, true);
       	
		if (!client->GetObject(key, "/tmp/" + key)) {
           	return 1;
        }
        
    }
    */
	std::string key = std::string(object_name.c_str()) + std::to_string(0).c_str();
   	client->PutObjectAsync(key, std::string(fname.c_str()), test_put_done, nullptr);

   	sleep(2);
	
/*
	Aws::String key = Aws::String("test_obj9991");
   	client->PutObject(key, fname);
	if (!client->GetObject(key, "/tmp/" + key)) {
       	return 1;
    }
*/       
    //dss::Objects *objs = client->GetObjects();
    //while (!objs->GetObjKeys()) {;}

	//auto result = client->ListObjects("");
	//for (auto k : result)
	//	printf("%s\n", k.c_str());

	//client.DeleteBucket(bucket_name, true);

    return 0;
}
