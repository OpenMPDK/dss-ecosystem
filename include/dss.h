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
#include <sys/stat.h>

#include <aws/core/Aws.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/auth/AWSCredentials.h>

#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>
#include <aws/s3/model/CreateBucketRequest.h>
#include <aws/s3/model/DeleteBucketRequest.h>

#include <aws/s3/model/BucketLocationConstraint.h>

#include "json.hpp"
#include "pr.h"

namespace dss {

using Credentials = Aws::Auth::AWSCredentials;
using Config = Aws::Client::ClientConfiguration;
using Callback = std::function<void(void*, std::string, std::string, int)>;

#define DSS_VER					"20210217"
#define DSS_PAGINATION_DEFAULT	100UL

class NoSuchResourceError : std::exception {
public:
    const char* what() const noexcept {return "Key doesn't exist\n";}
};

class DiscoverError : std::exception {
public:
	DiscoverError(Aws::String msg) {
		m_msg = "Discover failed: " + msg; 
	}
    const char* what() const noexcept { return m_msg.c_str(); }
private:
	Aws::String m_msg;
};

class NetworkError : std::exception {
public:
    const char* what() const noexcept {return "Endpoint cannot be reached\n";}
};

class GenericError : std::exception {
public:
    GenericError(std::string msg) : m_msg(msg) {}
    const char* what() const noexcept {return "Generic error\n";}
private:
    std::string m_msg;
};

/* Not using __attribute__((destructor)) b/c it is only called
   after global var is destructed, so if options is declared
   global, ShutdownAPI() would crash */
class DSSInit {
public:
	DSSInit(): m_options() 
	{
		char *c = NULL;
		unsigned l = 0;

		if ((c = getenv("DSS_AWS_LOG"))) {
			l = *c - '0';

			if (l > (int) Aws::Utils::Logging::LogLevel::Trace) {
				pr_err("AWS log level out of range\n");
				l = 0;
			}
		}

    	m_options.loggingOptions.logLevel = (Aws::Utils::Logging::LogLevel) l;
    	Aws::InitAPI(m_options);
	}

	~DSSInit() 
	{
		Aws::ShutdownAPI(m_options);
	}

private:
	Aws::SDKOptions m_options;
};

class Result;
class Client;
class ClusterMap;
class Cluster;

struct SesOptions {
	SesOptions()
	{
		scheme = "http";
    	useDualStack = false;
    	maxConnections = 25;
    	httpRequestTimeoutMs = 0;
    	requestTimeoutMs = 3000;
    	connectTimeoutMs = 1000;
    	enableTcpKeepAlive = true;
    	tcpKeepAliveIntervalMs = 30000;
	}

    std::string scheme;
    bool useDualStack;
    int maxConnections;
    int httpRequestTimeoutMs; // CURLOPT_TIMEOUT_MS
    int requestTimeoutMs;
    int connectTimeoutMs;
    int enableTcpKeepAlive;
    int tcpKeepAliveIntervalMs;

	static const std::string HTTP_SCHEME;
	static const std::string HTTPS_SCHEME;
};

const std::string SesOptions::HTTP_SCHEME = "http";
const std::string SesOptions::HTTPS_SCHEME = "https";

struct Request {
	typedef Result (Cluster::*Handler) (Request* r);

	Request(const char* k) :
		key(k) {}
	Request(const char* k, const char* f) :
		key(k),
		file(f) {}
	Request(const char* k, const char* f, Callback cb, void* cb_arg) :
		key(k),
		file(f),
		done_func(cb),
		done_arg(cb_arg) {}

	Result Submit(Handler h);

	const std::string	key;
	const std::string	file;
	Callback			done_func;
	void*				done_arg;
	uint32_t			key_hash;
	Cluster*			cluster;
	std::shared_ptr<Aws::IOStream> io_stream;
};

class CallbackCtx : public Aws::Client::AsyncCallerContext {
public:
	CallbackCtx(Callback func, void* args) {
		cb_func = func;
		cb_args = args;
	}

	Callback getCbFunc() const { return cb_func; }
	void* getCbArgs() const { return cb_args; } 

private:
	Callback cb_func;
	void* cb_args;
};

class Objects {
public:
    Objects(ClusterMap* map, std::string prefix, std::string delimiter, uint32_t ps) :
        m_cur_id(-1), m_cluster_map(map), m_prefix(prefix), m_delim(delimiter), m_pagesize(ps) {}
    const char *GetPrefix() { return m_prefix.c_str(); }
    std::string& GetDelim() { return m_delim; }
	uint32_t GetPageSize() { return m_pagesize; }
    int GetObjKeys();

	void SetToken(Aws::String str) { m_token = str; }
	Aws::String& GetToken() { return m_token; }
    bool TokenSet() { return m_token.size() != 0; }
    bool PageSizeSet() { return m_pagesize != 0; }
    std::set<std::string>& GetPage() { return m_page; }

private:
	int m_cur_id;
	Aws::String m_token;
   	bool m_token_set;
	ClusterMap* m_cluster_map;
	std::string m_prefix;
	std::string m_delim;
	uint32_t m_pagesize;
    std::set<std::string> m_page;
public:
    decltype(m_page.cbegin()) begin() const { return m_page.cbegin(); }
    decltype(m_page.cend()) end() const { return m_page.cend(); }
};

class Result {
public:
	Result() {}
	Result(bool success) : r_success(success) {}
	Result(bool success, Aws::S3::Model::GetObjectResult gor) :
			r_success(success), r_object(std::move(gor)) {}
	Result(bool success, Aws::S3::S3Error e) :
		r_success(success),
		r_err_type(e.GetErrorType()),
		r_err_msg("Exception: " + e.GetExceptionName() +
				  " Details: " + e.GetMessage()) {}
/*
	Result(Result& other) :
		r_success(other.r_success),
		r_err_type(other.r_err_type),
		r_err_msg(std::move(other.r_err_msg)),
		r_object(std::move(other.r_object))
	{}

	Result(Result&& other) :
		r_success(other.r_success),
		r_err_type(other.r_err_type),
		r_err_msg(std::move(other.r_err_msg)),
		r_object(std::move(other.r_object))
	{}
*/
	bool IsSuccess() { return r_success; }
	Aws::IOStream& GetIOStream() { return r_object.GetBody(); }
	Aws::S3::S3Errors GetErrorType() { return r_err_type; }
	Aws::String& GetErrorMsg() { return r_err_msg; }

private:
	bool				r_success;
	Aws::S3::S3Errors 	r_err_type;
	Aws::String			r_err_msg;
	Aws::S3::Model::GetObjectResult	r_object;
};

class Endpoint {
public:
	Endpoint(Credentials& cred, const std::string& url, Config& cfg);

	Result GetObject(const Aws::String& bn, Request* req);
    Result GetObject(const Aws::String& bn, const Aws::String& objectName);
 	Result PutObjectAsync(const Aws::String& bn, Request* req);
	Result PutObject(const Aws::String& bn, Request* req);
    Result PutObject(const Aws::String& bn, const Aws::String& objectName, std::shared_ptr<Aws::IOStream>& input_stream);
	Result DeleteObject(const Aws::String& bn, Request* req);
    Result DeleteObject(const Aws::String& bn, const Aws::String& objectName);

	Result HeadBucket(const Aws::String& bn);
	Result CreateBucket(const Aws::String& bn);

	Result ListObjects(const Aws::String& bn, Objects *objs);

private:
	Aws::S3::S3Client m_ses;
};

class Cluster {
public:
	Cluster(uint32_t id) :
		m_id(id),
		m_bucket(Aws::String(DATA_BUCKET_PREFIX) + Aws::String(std::to_string(id).c_str())) {}

	~Cluster()
	{
		for (auto e : m_endpoints)
			delete e;
	}

	Endpoint* GetEndpoint(Request* r) { return m_endpoints[r->key_hash % m_endpoints.size()]; }

    Result GetObject(const Aws::String& objectName);
    Result PutObject(const Aws::String& objectName, std::shared_ptr<Aws::IOStream>& input_stream);
    Result DeleteObject(const Aws::String& objectName);

    Result GetObject(Request* r);
	Result PutObjectAsync(Request* r);
    Result PutObject(Request* r);
    Result DeleteObject(Request* r);
 
    Result HeadBucket();
    Result HeadBucket(const Aws::String& bucketName);

	Result CreateBucket();
    uint32_t GetID() { return m_id; }

	Result ListObjects(Objects *objs);

	int InsertEndpoint(Client* c, const std::string& ip, uint32_t port);
private:
	uint32_t m_id;
	Aws::String m_bucket;
   	std::vector<Endpoint*> m_endpoints;

	static constexpr char* DATA_BUCKET_PREFIX = (char*)"dss";
};

class ClusterMap {
public:
	ClusterMap(Client *c) : m_client(c) {}

	~ClusterMap()
	{
		for (auto c : m_clusters)
			delete c;
	}

	Cluster* InsertCluster(uint32_t id)
	{
		Cluster* c = new Cluster(id);
		if (m_clusters.size() < (id + 1))
			m_clusters.resize(id + 1);
		m_clusters.at(id) = c;

		return c;
	}


	void GetCluster(Request* req);
	int DownloadClusterConf();
	int VerifyClusterConf();

	const std::vector<Cluster*>& GetClusters() { return m_clusters; }
/*
	unsigned GetCLWeight(unsigned i, Aws::Utils::ByteBuffer hash_bb)
	{
		static size_t cl_base = EP_SLOT_WIDTH / 8;

		if (hash_bb.GetLength() != 256/8)
			abort();

		if (i >= CL_SLOT_MAX)
			throw std::out_of_range("Cluster hash table");

		uint8_t *hash = hash_bb.GetUnderlyingData();

		return (hash[cl_base + (i/2)] >> ((i%2) * CL_SLOT_BITS)) & CL_SLOT_MASK;
	}
*/
	unsigned GetCLWeight(unsigned i, const Aws::String& key)
	{
		return m_hash(std::to_string(i) + std::string(key.c_str()));
	}

	unsigned GetCLWeight(unsigned i, char* key)
	{
		return m_hash(std::to_string(i) + std::string(key));
	}

	unsigned GetEPWeight(unsigned i);
	
private:
	Client* m_client;
	std::hash<std::string> m_hash;
	std::vector<Cluster*> m_clusters;
#if 0
	static const uint64_t SLOT_TBL_SIZE = 8;
	static const uint64_t EP_SLOT_BITS = 5; /**/
	static const uint64_t EP_SLOT_MAX =  1ULL << EP_SLOT_BITS;
	static const uint64_t EP_SLOT_MASK = (1ULL << EP_SLOT_BITS) - 1;
	static const uint64_t EP_SLOT_WIDTH = EP_SLOT_MAX * EP_SLOT_BITS;

	static const uint64_t CL_SLOT_BITS = 4;
	static const uint64_t CL_SLOT_MAX = 1ULL << CL_SLOT_BITS;
	static const uint64_t CL_SLOT_MASK = (1ULL << CL_SLOT_BITS) - 1;
#endif	
}; 

class Client {
public:
	~Client();

    Result GetClusterConfig();
	int InitClusterMap();
	static std::unique_ptr<Client> CreateClient(const std::string& url, const std::string& user,
												const std::string& pwd, const SesOptions& opts = SesOptions());
	
	Config ExtractOptions(const SesOptions& opts);
	Credentials& GetCredential() { return m_cred; }
	Config& GetConfig() { return m_cfg; }

    int GetObject(const Aws::String& objectName, const Aws::String& dest_fn);
    int PutObject(const Aws::String& objectName, const Aws::String& src_fn, bool async = false);
	int PutObjectAsync(const std::string& objectName, const std::string& src_fn,
					   Callback cb = [](void* ptr, std::string key, std::string message, int err){}, void *cb_arg = nullptr);

    int DeleteObject(const Aws::String& objectName);

    std::unique_ptr<Objects> GetObjects(std::string prefix, std::string delimiter, uint32_t page_size = DSS_PAGINATION_DEFAULT) {
    	return std::unique_ptr<Objects>(new Objects(m_cluster_map, prefix, delimiter, page_size));
    };
    std::set<std::string>&& ListObjects(const std::string& prefix, const std::string& delimiter);
    std::set<std::string> ListBuckets();

private:
	Client(const std::string& url, const std::string& user, const std::string& pwd,
			const SesOptions& opts) {
		m_cfg = ExtractOptions(opts);
		m_cred = Aws::Auth::AWSCredentials(user.c_str(), pwd.c_str());
		m_discover_ep = new Endpoint(m_cred, url, m_cfg);
	}

    friend class Objects;
	Credentials m_cred;
	Config m_cfg;	

    Endpoint* m_discover_ep;
    ClusterMap* m_cluster_map;

    static constexpr char* DISCOVER_BUCKET = (char *)"dss";
    static constexpr char* DISCOVER_CONFIG_KEY = (char *)"conf.json";
};

}; //namespace dss
