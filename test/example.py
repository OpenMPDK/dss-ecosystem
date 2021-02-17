import sys
import io

sys.path.insert(0, '/root/jerry/dss_client/build')

import dss
import os
import trio


access_key = "minioadmin"
access_secret = "minioadmin"
discover_endpoint = 'http://127.0.0.1:9001'

def main():
	# Provide non-default options
	option = dss.clientOption();
	option.maxConnections = 1;

	# Create a client session against minio cluster(s)
	# It could fail b/c:
	#	1) network unreachable	
	#	2) config file 'conf.json' or containing bucket 'bss' is missing
	try:
		client = dss.createClient(discover_endpoint, access_key,
								  access_secret, option)
	except Exception as e:
		print(e)
		return None

	# Upload this script to cluster under key name exampleXX
	key_base = 'example'
	filename = os.path.abspath(__file__)
	for i in range(20):
		key = key_base + str(i)
		client.putObject(key, filename)

	# Print objects with prefix
	objects = client.listObjects(key_base + str(1))
	for o in objects:
		print(o)

	# Retrieve object then delete
	for i in range(20):
		key = key_base + str(i)
		client.getObject(key, '/tmp/' + key)
		client.deleteObject(key)

if __name__ == "__main__":
	main()
