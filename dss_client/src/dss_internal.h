/**
  The Clear BSD License

  Copyright (c) 2023 Samsung Electronics Co., Ltd.
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted (subject to the limitations in the disclaimer
  below) provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
 * Neither the name of Samsung Electronics Co., Ltd. nor the names of its
 contributors may be used to endorse or promote products derived from this
 software without specific prior written permission.
 NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **/

#ifndef DSS_INTERNAL_H
#define DSS_INTERNAL_H

#include "pr.h"
#include "rdd_cl.h"
#include <unordered_map>
#include <mutex>

namespace dss {

	class Client;
	class Cluster;

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

	struct Request {
		typedef Result (Cluster::*Handler) (Request* r);
		typedef Result (Cluster::*Handler_with_buffer) (Request* r, unsigned char* resp_buff, long long buffer_size);


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
		Result Submit_with_buffer(Handler_with_buffer handler, unsigned char* resp_buff, long long buffer_size);

		const std::string	key;
		const std::string	file;
		Callback			done_func;
		void*				done_arg;
		uint32_t			key_hash;
		Cluster*			cluster;
		std::shared_ptr<Aws::IOStream> io_stream;
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
			Result(bool success, long long content_length):
				r_success(success), r_content_length(content_length) {}

			bool IsSuccess() { return r_success; }
			Aws::IOStream& GetIOStream() { return r_object.GetBody(); }
			long long GetContentLength() { return r_object.GetContentLength();}
			long long GetContentLengthValue() { return r_content_length;}
			Aws::S3::S3Errors GetErrorType() { return r_err_type; }
			Aws::String& GetErrorMsg() { return r_err_msg; }

		private:
			bool				r_success;
			Aws::S3::S3Errors 	r_err_type;
			Aws::String			r_err_msg;
			long long           r_content_length;
			Aws::S3::Model::GetObjectResult	r_object;
	};

	class Endpoint {
		public:
			Endpoint(Aws::Auth::AWSCredentials& cred, const std::string& url, Config& cfg, const std::string transport_type, const std::string uuid);

			Result GetObject(const Aws::String& bn, Request* req);
			Result GetObject(const Aws::String& bn, const Aws::String& objectName);
			Result GetObject(const Aws::String& bn, Request* req, unsigned char* res_buff, long long buffer_size);
			Result PutObject(const Aws::String& bn, Request* req, unsigned char* res_buff, long long buffer_size);

			Result GetObjectAsync(const Aws::String& bn, Request* req);
			Result PutObject(const Aws::String& bn, Request* req);
			Result PutObject(const Aws::String& bn, const Aws::String& objectName, std::shared_ptr<Aws::IOStream>& input_stream);
			Result PutObjectAsync(const Aws::String& bn, Request* req);
			Result DeleteObject(const Aws::String& bn, Request* req);
			Result DeleteObject(const Aws::String& bn, const Aws::String& objectName);

			Result HeadBucket(const Aws::String& bn);
			Result CreateBucket(const Aws::String& bn);
			Result DeleteBucket(const Aws::String& bn);

			Result ListObjects(const Aws::String& bn, Objects *objs);

		private:
			Aws::S3::S3Client m_ses;
			std::string m_uuid;
			std::string m_transport_type;
	};

	class RDDEndpoint {
		public:
			RDDEndpoint(const std::string& rdd_ip, uint32_t rdd_port, const std::string uuid);

			int GetRddHandleAndPort(rdd_cl_conn_ctx_t *rdd_conn, std::string &rdd_ip, uint32_t *rdd_port){
				*rdd_conn = m_rdd_conn;
				rdd_ip = m_rdd_ip;
				*rdd_port = m_rdd_port;
				return 0;
			};

			int get_rkey(void* buff, uint32_t* rkey) {
				int ret = 0;
				pr_debug("RDDEP: get_rkey before");
				std::unique_lock<std::mutex> ul(rkey_lock);
				auto search = m_rkey_map.find((uint64_t)buff);
				ul.unlock();
				if (search != m_rkey_map.end()) {
					*rkey = search->second;
					pr_debug("RDDEP: rkey found for buff %lu", (uint64_t)buff);
				} else {
					ret = -1;
				} 				
				return ret;
			};

			int create_rkey(void* buff, uint64_t buff_len, rdd_cl_conn_ctx_t *rdd_conn, uint32_t* rkey) {
				pr_debug("RDDEP: create_rkey before");
				struct ibv_mr *mr = rdd_cl_conn_get_mr(rdd_conn, buff, buff_len);
				pr_debug("RDDEP: qhandle %x\n", rdd_conn->qhandle);
				*rkey = mr->rkey;
				pr_debug("RDDEP: create_rkey after - %u", *rkey);
				std::unique_lock<std::mutex> ul(rkey_lock);
				m_rkey_map[(uint64_t)buff] = *rkey;
				return 0;
			};



		private:
			std::string m_uuid;
			rdd_cl_conn_ctx_t m_rdd_conn;
			std::string m_rdd_ip;
			uint32_t m_rdd_port;
			std::unordered_map<uint64_t, uint32_t> m_rkey_map = {};
			std::mutex rkey_lock; 

	};

	class Cluster {
		public:
			Cluster(uint32_t id, const std::string& instance_uuid) :
				m_id(id),
				m_uuid(instance_uuid),
				m_bucket(Aws::String(DATA_BUCKET_PREFIX) + Aws::String(std::to_string(id).c_str())),
				m_instance_uuid(instance_uuid) {}

			~Cluster()
			{
				for (auto e : m_endpoints)
					delete e;
			}

			Endpoint* GetEndpoint(Request* r) { return m_endpoints[r->key_hash % m_endpoints.size()]; }
			Endpoint* GetEndpoint(std::size_t key_hash) { return m_endpoints[key_hash % m_endpoints.size()]; }

			Result GetObject(const Aws::String& objectName);
			Result GetObject(Request* req, unsigned char* res_buff, long long buffer_size);
			Result PutObject(Request* req, unsigned char* res_buff, long long buffer_size);

			Result PutObject(const Aws::String& objectName, std::shared_ptr<Aws::IOStream>& input_stream);
			Result DeleteObject(const Aws::String& objectName);

			Result GetObject(Request* r);
			Result GetObjectAsync(Request* r);
			Result PutObject(Request* r);
			Result PutObjectAsync(Request* r);
			Result DeleteObject(Request* r);

			Result HeadBucket();
			Result HeadBucket(const Aws::String& bucketName);

			Result CreateBucket();
			uint32_t GetID() { return m_id; }

			Result ListObjects(Objects *objs);
			int InsertEndpoint(Client* c, const std::string& ip, uint32_t port, const std::string transport_type, const std::string uuid);

			int CreateRDDConnection(const std::string& rdd_ip, uint32_t rdd_port, const std::string uuid);
			int GetRddKeyFormat(std::string &key, std::string &value){
				rdd_cl_conn_ctx_t rdd_conn;
				uint32_t rdd_port;
				std::string rdd_ip;
				key = std::string(m_uuid) + std::string(".dss.rdd.init");
				for (auto e: m_rdd_endpoints){
					e->GetRddHandleAndPort(&rdd_conn, rdd_ip, &rdd_port);
					value += rdd_ip + std::string("-") + std::to_string(rdd_port) + std::string("::");
					value += std::to_string(int(rdd_conn.qhandle)) + std::string("##");
				}
				value += std::string("END##");
				return 0;
			}

			int GetOneRDDConnection(rdd_cl_conn_ctx_t *rdd_conn, unsigned char* res_buff, long long buffer_size, uint32_t *rkey){
				//printf("GetOneRDD func called\n");
				RDDEndpoint *ep = m_rdd_endpoints.back();
				rdd_cl_conn_ctx_t t_rdd_conn;
				std::string rdd_ip;
				uint32_t rdd_port;
				uint32_t t_rKey;
				ep->GetRddHandleAndPort(&t_rdd_conn, rdd_ip, &rdd_port);
				if (ep->get_rkey(res_buff, &t_rKey) != 0) {
					ep->create_rkey(res_buff, buffer_size, &t_rdd_conn, &t_rKey);
				}
				std::memcpy(rdd_conn, &t_rdd_conn, sizeof(rdd_cl_conn_ctx_t));
				*rkey = t_rKey;
				//printf("GetOneRDD IP %d, %x, %s\n", rdd_conn->conn_id, rdd_conn->qhandle, rdd_ip.c_str());
				return 0;
			}
			std::vector<Endpoint*> GetEndpoints(){ return m_endpoints; }
			Aws::String GetBucket(){ return m_bucket;}

		private:
			uint32_t m_id;
			std::string m_uuid;
			Aws::String m_bucket;
			std::vector<Endpoint*> m_endpoints;
			std::vector<RDDEndpoint*> m_rdd_endpoints;

			static constexpr char* DATA_BUCKET_PREFIX = (char*)"dss";
			std::string m_instance_uuid;
	};


	/* Not using __attribute__((destructor)) b/c it is only called
	 * after global var is destructed, so if options is declared
	 * global, ShutdownAPI() would crash */
	class DSSInit {
		public:
			DSSInit(): m_local_config(nullptr), m_options()
		{
			char *s = NULL;
			unsigned l = 0;

			if ((s = getenv("DSS_AWS_LOG"))) {
				l = *s - '0';

				if (l > (int) Aws::Utils::Logging::LogLevel::Trace) {
					pr_err("AWS log level out of range\n");
					l = 0;
				}
				m_options.loggingOptions.logLevel = (Aws::Utils::Logging::LogLevel) l;
			}

			if ((s = getenv("DSS_AWS_LOG_FILENAME")))
				m_options.loggingOptions.defaultLogPrefix = s;

			if ((s = getenv("DSS_CONFIG_FILE")))
				m_local_config = s;

			s = (char*) "AWS_EC2_METADATA_DISABLED=true";
			if (putenv(s))
				pr_err("Failed to set AWS_EC2_METADATA_DISABLED\n");

			Aws::InitAPI(m_options);
		}

			~DSSInit()
			{
				Aws::ShutdownAPI(m_options);
			}

			const char* GetConfPath() { return m_local_config; }
			std::mutex& mutex() { return m_mutex; }

		private:
			std::mutex m_mutex;
			const char* m_local_config;
			Aws::SDKOptions m_options;
	};

	class ClusterMap {
		private:
			enum class Status : int {
				EMPTY,
				ALL_GOOD,
				PARTIAL
			};
			enum class State : int {
				CREATE,
				TEST,
				EXIT
			};

		public:
			ClusterMap(Client *c, DSSInit& i) :
				m_wait_time(3), m_client(c), m_init(i) {}

			~ClusterMap()
			{
				for (auto c : m_clusters)
					delete c;
			}

			Cluster* InsertCluster(uint32_t id, const std::string& instance_uuid)
			{
				Cluster* c = new Cluster(id, instance_uuid);
				if (m_clusters.size() < (id + 1))
					m_clusters.resize(id + 1);
				m_clusters.at(id) = c;

				return c;
			}

			void GetCluster(Request* req);
			const char* GetClusterConfFromLocal() { return m_init.GetConfPath(); }
			int AcquireClusterConf(const std::string& uuid, const unsigned int endpoints_per_cluster);
			int VerifyClusterConf();
			Status DetectClusterBuckets(bool);
			Result TryLockClusters();
			Result UnlockClusters();

			const std::vector<Cluster*>& GetClusters() { return m_clusters; }

			unsigned GetCLWeight(unsigned i, const Aws::String& key)
			{
				return m_hash(std::to_string(i) + std::string(key.c_str()));
			}

			unsigned GetCLWeight(unsigned i, char* key)
			{
				return m_hash(std::to_string(i) + std::string(key));
			}

			unsigned GetCLWeight(const std::string key1, const std::string key2)
			{
				return m_hash(key1 + key2);
			}

			unsigned GetEPWeight(unsigned i);

		private:
			unsigned m_wait_time;
			Client* m_client;
			DSSInit& m_init;
			std::hash<std::string> m_hash;
			std::vector<Cluster*> m_clusters;
	};
}

#endif // DSS_INTERNAL_H

