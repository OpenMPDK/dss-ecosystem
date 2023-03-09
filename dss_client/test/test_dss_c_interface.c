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

int main() {
	DSSClient *c;
	int fd = -1;
	int ret = -1;
	int size = 1024*1024;
	unsigned char* buff = (unsigned char*)calloc(1, size);
	unsigned char* buff1 = (unsigned char*)calloc(1, size);
	char obj_name[256];

	strcpy(obj_name, "testfile1");
	
	fd = open("/dev/urandom", O_RDONLY);
	if (read(fd, buff, size) != size) {
		printf("Failed to read random data\n");
		goto out;
	}
    char endpoint[] = "206.0.0.214:9000";
    char access[] = "minio";
    char secret[] = "minio123";
    char uuid[] = "12345";
	c = (DSSClient *)DSSClientInit(endpoint, access, secret, uuid, 256);
	if (c == NULL) {
		printf("Client init failed\n");
		goto out;
	}		
	ret = PutObjectBuffer(c, (void*) obj_name, strlen(obj_name), buff, size);
	printf("Object testfile1 uploaded. return value: %d\n", ret);
	
	ret = GetObjectBuffer(c, (void*) obj_name, strlen(obj_name), buff1, size);
	printf("Object testfile1 downloaded. return value: %d\n", ret);
	//printf("Buffer Size <%d> Content  <%s>\n", ret, buff);
	if (memcmp(buff, buff1, size) != 0) {
		printf("Invalid data [%x] -> [%x]\n", buff, buff1);
		hexdump(buff, size);
		hexdump(buff1, size);
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
