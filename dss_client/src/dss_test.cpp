#include <unistd.h>

#include "dss.h"

	void
test_put_done(void* ptr, std::string key, std::string message, int err)
{
	printf("%s: key %s message %s\n", __func__, key.c_str(), message.c_str());
}

	void
test_get_done(void* ptr, std::string key, std::string message, int err)
{
	printf("%s: key %s message %s\n", __func__, key.c_str(), message.c_str());
}

	void*
client_creation_task(void*)
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
	const std::string object_name = "test_obj";
	const std::string fname = "/root/jerry/dss_client/src/dss_client.cpp";
	const std::string tmp_fname = "/tmp/async_get.dat";

	// Test for simultaneous client creation
	for (int i=0; i<5; i++) {
		pthread_t t;
		pthread_create(&t, NULL, client_creation_task, NULL);
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
	std::string key = object_name + std::to_string(0).c_str();
	client->PutObjectAsync(key, fname, test_put_done, nullptr);
	client->GetObjectAsync(key, tmp_fname, test_get_done, nullptr);

	sleep(2);

	return 0;
}

