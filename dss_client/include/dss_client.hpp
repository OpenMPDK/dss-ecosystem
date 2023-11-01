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

#ifndef DSS_CLIENT_HPP
#define DSS_CLIENT_HPP

#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/s3/S3Client.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace dss {
	namespace py = pybind11;

#define DSS_VER    "71d317f"
#define DSS_PAGINATION_DEFAULT	100UL

	class Endpoint;
	class Result;
	class ClusterMap;

	using Credentials = Aws::Auth::AWSCredentials;
	using Config = Aws::Client::ClientConfiguration;
	using Callback = std::function<void(void*, std::string, std::string, int)>;

	class NoSuchResourceError : std::exception {
		public:
			NoSuchResourceError() : m_msg("Key doesn't exist") {}
			const char* what() const noexcept { return m_msg.c_str(); }
		private:
			std::string m_msg;
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
			NetworkError() : m_msg() {}
			NetworkError(std::string msg) : m_msg(std::move(msg)) {}
			const char* what() const noexcept { return m_msg.c_str(); }
		private:
			std::string m_msg;
	};

	class FileIOError : std::exception {
		public:
			FileIOError() : m_msg() {}
			FileIOError(std::string msg) : m_msg(std::move(msg)) {}
			const char* what() const noexcept { return m_msg.c_str(); }
		private:
			std::string m_msg;
	};

	class NewClientError : std::exception {
		public:
			NewClientError() : m_msg() {}
			NewClientError(std::string msg) : m_msg(std::move(msg)) {}
			const char* what() const noexcept { return m_msg.c_str(); }
		private:
			std::string m_msg;
	};

	class GenericError : std::exception {
		public:
			GenericError(std::string msg) : m_msg(std::move(msg)) {}
			const char* what() const noexcept { return m_msg.c_str(); }
		private:
			std::string m_msg;
	};

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
	};

	class Objects {
		public:
			Objects(ClusterMap* map, std::string prefix, std::string delimiter, bool cp, uint32_t ps) :
				m_cur_id(-1),
				m_cluster_map(map),
				m_prefix(prefix),
				m_delim(delimiter),
				m_comm_prefix(cp),
				m_pagesize(ps) {}
			const char *GetPrefix() { return m_prefix.c_str(); }
			std::string& GetDelim() { return m_delim; }
			uint32_t GetPageSize() { return m_pagesize; }
			int GetObjKeys();

			void SetToken(Aws::String str) { m_token = str; }
			Aws::String& GetToken() { return m_token; }
			bool TokenSet() { return m_token.size() != 0; }
			bool NeedCommPrefix() { return m_comm_prefix; }
			bool PageSizeSet() { return m_pagesize != 0; }
			std::set<std::string>& GetPage() { return m_page; }
			std::set<std::string>& GetCPre() { return m_cps;}

		private:
			int m_cur_id;
			Aws::String m_token;
			bool m_token_set;
			ClusterMap* m_cluster_map;
			std::string m_prefix;
			std::string m_delim;
			bool m_comm_prefix;
			uint32_t m_pagesize;
			std::set<std::string> m_page;
			std::set<std::string> m_cps;
		public:
			decltype(m_page.cbegin()) begin() const { return m_page.cbegin(); }
			decltype(m_page.cend()) end() const { return m_page.cend(); }
	};

	class Client {
		public:
			Client(const std::string& url, const std::string& user, const std::string& pwd,
					const SesOptions& opts, const std::string& uuid);
			~Client();

			Result GetClusterConfig();
			int InitClusterMap(const std::string& uuid, const unsigned int max_endpoints);
			Result TryLockClusters();
			Result UnlockClusters();
			static std::unique_ptr<Client> CreateClient(const std::string& url,
					const std::string& user,
					const std::string& pwd,
					const SesOptions& opts = SesOptions(),
					const std::string& uuid = "644bb326-019a-4443-b4ef-0d5eb0e57914",
					const unsigned int max_endpoints = 255);

			Config ExtractOptions(const SesOptions& opts);
			Credentials& GetCredential() { return m_cred; }
			Config& GetConfig() { return m_cfg; }

			int GetObject(const Aws::String& objectName, const Aws::String& dest_fn);
			PYBIND11_EXPORT int GetObjectNumpyBuffer(const Aws::String& objectName, py::array_t<uint8_t> numpy_buffer);
			PYBIND11_EXPORT int GetObjectBufferPython(const Aws::String& objectName, py::buffer buffer);
			int GetObjectBuffer(const Aws::String& objectName, unsigned char* buffer, long int buffer_size);
			int GetObjectAsync(const std::string& objectName, const std::string& dst_fn,
					Callback cb, void* cb_arg);
			int PutObject(const Aws::String& objectName, const Aws::String& src_fn, bool async = false);
			int PutObjectAsync(const std::string& objectName, const std::string& src_fn,
					Callback cb = [](void* ptr, std::string key, std::string message, int err){},
					void *cb_arg = nullptr);
			PYBIND11_EXPORT int PutObjectBufferPython(const Aws::String& objectName, py::buffer buffer, int content_length);
			int PutObjectBuffer(const Aws::String& objectName, unsigned char* buffer, int content_length);
			int DeleteObject(const Aws::String& objectName);
			std::unique_ptr<Objects> GetObjects(std::string prefix, std::string delimiter,
					bool comm_prefix = false,
					uint32_t page_size = DSS_PAGINATION_DEFAULT);
			std::set<std::string>&& ListObjects(const std::string& prefix, const std::string& delimiter);
			std::set<std::string> ListBuckets();

		private:

			friend class Objects;
			Credentials m_cred;
			Config m_cfg;	

			Endpoint* m_discover_ep;
			ClusterMap* m_cluster_map;

			// Bucket names can consist only of lowercase letters, numbers, dots (.), and hyphens
			static constexpr char* LOCK_BUCKET = (char *)"dss-lock";
			static constexpr char* DISCOVER_BUCKET = (char *)"dss";
			static constexpr char* DISCOVER_CONFIG_KEY = (char *)"conf.json";
	};

}; //namespace dss

#endif /* DSS_CLIENT_HPP */

