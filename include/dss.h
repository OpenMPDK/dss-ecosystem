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

using Credential = Aws::Auth::AWSCredentials;

extern volatile bool dss_env_init;

struct Config {
	std::string ip;
	std::string user;
	std::string& pwd;
};

class NoSuchResouceError : std::exception {
public:
    const char* what() const noexcept {return "Key doesn't exist\n";}
};

class NetworkError : std::exception {
public:
    const char* what() const noexcept {return "Endpoint cannot be reached\n";}
};

class GenericError : std::exception {
public:
    GenericError(std::string msg) : m_msg(msg) {}
    const char* what() const noexcept {return "Endpoint cannot be reached\n";}
private:
    std::string m_msg;
};

int InitAwsAPI();
int FiniAwsAPI();

class Client;
class ClusterMap;

class Objects {
public:
    Objects(ClusterMap* map) :
        m_cur_id(-1), m_cluster_map(map)  {}
    int GetObjKeys();
private:
	int m_cur_id;
	std::string m_token;
   	bool m_token_set;

	ClusterMap* m_cluster_map;
    std::set<std::string> m_pages;
public:
    decltype(m_pages.cbegin()) begin() const { return m_pages.cbegin(); }
    decltype(m_pages.cend()) end() const { return m_pages.cend(); }
};

class Result {
public:
	Result(bool success) : r_success(success) {}
	Result(bool success, Aws::S3::Model::GetObjectResult gor) :
			r_success(success), r_object(std::move(gor)) {}
	Result(bool success, Aws::S3::S3Error e) :
		r_success(success),
		r_err_type(e.GetErrorType()),
		r_err_msg(e.GetExceptionName() + e.GetMessage()) {}

	bool IsSuccess() { return r_success; }
	Aws::IOStream& GetIOStream() { return r_object.GetBody(); }
	Aws::S3::S3Errors GetErrorType() { return r_err_type; }
	Aws::String& GetErrorMsg() { return r_err_msg; }

private:
	bool			r_success;
	Aws::S3::S3Errors r_err_type;
	Aws::String		r_err_msg;
	Aws::S3::Model::GetObjectResult	r_object;
};

class Endpoint {
public:
	Endpoint(Credential& cred, const std::string& url);

    Result GetObject(const Aws::String& bn, const Aws::String& objectName);
    Result PutObject(const Aws::String& bn, const Aws::String& objectName, std::shared_ptr<Aws::IOStream>& input_stream);
    Result DeleteObject(const Aws::String& bn, const Aws::String& objectName);

	Result HeadBucket(const Aws::String& bn);
	Result CreateBucket(const Aws::String& bn);

	Result ListObjects(const Aws::String& bn, const Aws::String& prefix, std::set<std::string>& keys);

#if 0
    int CreateBucket(const Aws::String& bucketName);
	int DeleteBucket(const Aws::String& bucketName, bool force=false);
    Objects *GetObjects(std::string bucket) { return new Objects(this, bucket); };

    std::set<std::string> ListObjects(const Aws::String& bucketName);
    std::set<std::string> ListBuckets();

    Aws::S3::S3Client* GetAwsClient() { return m_client; }; 
#endif
private:
	Aws::S3::S3Client m_ses;
};

class Cluster {
public:
	Cluster(uint32_t id) : m_id(id), m_endpoints() {
		m_bucket = Aws::String(DATA_BUCKET_PREFIX) + Aws::String(std::to_string(id).c_str());
	}

    Result GetObject(const Aws::String& objectName);
    Result PutObject(const Aws::String& objectName, std::shared_ptr<Aws::IOStream>& input_stream);

    Result DeleteObject(const Aws::String& objectName);
    Result HeadBucket();
    Result HeadBucket(const Aws::String& bucketName);

	Result CreateBucket();
    uint32_t GetID() { return m_id; }

	Result ListObjects(const Aws::String& prefix, std::set<std::string>& keys);

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

	Cluster* InsertCluster(uint32_t id)
	{
		Cluster* c = new Cluster(id);
		if (m_clusters.size() < (id + 1))
			m_clusters.resize(id + 1);
		m_clusters.at(id) = c;

		return c;
	}

	Cluster* GetCluster(const Aws::String& key);
	int DownloadClusterConf();
	int VerifyClusterConf();

	const std::vector<Cluster*>& GetClusters() { return m_clusters; }
	unsigned GetCLWeight(unsigned i, uint64_t v)
	{
		Hash* h = (Hash*)&v;

		if (i >= MAX_SLOTS)
			throw std::out_of_range("Cluster hash table");

		return h->cl_slots & (SLOT_MASK << i);
	}

	unsigned GetEPWeight(unsigned i);
	
private:
	Client* m_client;
	std::hash<std::string> m_hash;
	std::vector<Cluster*> m_clusters;

	struct Hash {
		uint32_t ep_slots;
		uint32_t cl_slots;
	};

	static const uint64_t SLOT_BITS = 4;
	static const uint64_t SLOT_MASK = (1ULL << SLOT_BITS) - 1;
	static const uint64_t MAX_SLOTS = sizeof (Hash::ep_slots) * 8 / SLOT_BITS;
}; 

class Client {
public:
	Client(const std::string& ip, const std::string& user, const std::string& pwd) {
		if (__sync_bool_compare_and_swap(&dss_env_init, false, true))
			InitAwsAPI();

		m_cred = Aws::Auth::AWSCredentials(user.c_str(), pwd.c_str());
		m_discover_ep = new Endpoint(m_cred, ip);
	}

    Result GetClusterConfig();
	int InitClusterMap();
	static Client* CreateClient(const std::string& ip,
								const std::string& user, const std::string& pwd);

	Credential& GetCredential() { return m_cred; }

    int GetObject(const Aws::String& objectName, const Aws::String& dest_fn);
    int PutObject(const Aws::String& objectName, const Aws::String& src_fn);
    int DeleteObject(const Aws::String& objectName);

    Objects *GetObjects() { return new Objects(m_cluster_map); };
    std::set<std::string> ListObjects(const Aws::String& prefix);
    std::set<std::string> ListBuckets();

private:
    friend class Objects;
	Aws::Auth::AWSCredentials m_cred;
    Endpoint* m_discover_ep;
    ClusterMap* m_cluster_map;

	/* Explicit cast is mandatory */
    static constexpr char* DISCOVER_BUCKET = (char *)"dss";
    static constexpr char* DISCOVER_CONFIG_KEY = (char *)"conf.json";
};

}; //namespace dss
