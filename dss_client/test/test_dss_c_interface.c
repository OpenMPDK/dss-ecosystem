#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#include "dss.h"

int hexdump(unsigned char* buff, int size)
{
	int i = 0;
	for (i = 0; i < size; i++) {
		printf("%x", buff[i]);
	}
	printf("\n");
	return 0;
}

int main(int argc, char* argv[]) {
	DSSClient *c;
	int fd = -1;
	int ret = -1;
	int size = 1024*1024;
	unsigned char* buff;
	unsigned char* buff1;
	char obj_name[256];
    	char uuid[] = "12345";

	if (argc < 4){
		printf("Invalid number of arguments given\n");
		printf("Usage: %s <endpoint_url> <access_key> <secret_key> <size_in_kb(default 1MB)>\n");
		printf("Endpoint URL format is '<minio_host_name_or_ip>:<port>'\n");
		return -1;
	}
	if (strchr(argv[1], ':') == NULL) {
		printf("Invalid endpoint URL\n");
		return -1;
	}
	if (argc == 5)
		size = atoi(argv[4]) * 1024;
	buff = (unsigned char*)calloc(1, size);
	buff1 = (unsigned char*)calloc(1, size);

	strcpy(obj_name, "testfile1");
	
	fd = open("/dev/urandom", O_RDONLY);
	if (read(fd, buff, size) != size) {
		printf("Failed to read random data\n");
		goto out;
	}

	c = (DSSClient *)DSSClientInit(argv[1], argv[2], argv[3], uuid, 256);
	if (c == NULL) {
		printf("Client init failed\n");
		goto out;
	}		
	ret = PutObjectBuffer(c, (void*) obj_name, strlen(obj_name), buff, size);
	printf("Object testfile1 uploaded. return value: %d\n", ret);
	
	ret = GetObjectBuffer(c, (void*) obj_name, strlen(obj_name), buff1, size);
	printf("Object testfile1 downloaded. return value: %d\n", ret);
	//printf("Buffer Size <%d> Content  <%s>\n", ret, buff);
	if (ret < 0) 
		goto out;

	if (memcmp(buff, buff1, size) != 0) {
		printf("Invalid data [%x] -> [%x]\n", buff, buff1);
		printf("Printing partial data of 128 bytes of the data sent and received");
		hexdump(buff, 128);
		hexdump(buff1, 128);
		goto out;
	}

	printf("Object testfile1 verified\n");
	ret = DeleteObject(c, (void*) obj_name, strlen(obj_name));
	printf("Object testfile1 deleted\n");

out:
	if (fd != -1)
		close(fd);
	free(buff);
	free(buff1);
	return ret;
}
