import dss
import os
import time

access_key = "minioadmin"
access_secret = "minioadmin"
discover_endpoint = 'http://127.0.0.1:9001'

def callback(ctx):
	print(ctx.key)

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
	ctx = dss.asyncCtx();
	ctx.done_func = callback;
	ctx.done_arg = {1, 2, 3};

	#for i in range(2):
	key = key_base + str(0)
	client.putObjectAsync(key, filename, ctx)

	time.sleep(5)

"""
	# Print objects with prefix
	objects = client.getObjects("", "/", 8)
	while True:
		try:
			it = iter(objects)
		except dss.NoIterator:
			break
		while True:
			try:
				key = next(it)
				print("{}".format(key))
			except StopIteration:
				break

	# Retrieve object then delete
	for i in range(1):
		key = key_base + str(i)
		client.getObject(key, '/tmp/' + key)
		#client.deleteObject(key)
"""
if __name__ == "__main__":
	main()
