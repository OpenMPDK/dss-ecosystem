#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>

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

int putRecursive(DSSClient c, char *basePath)
{
    int i;
    char path[1000];
    struct dirent *dp;
    DIR *dir = opendir(basePath);

    if (!dir){
		return 0;
	}

    while ((dp = readdir(dir)) != NULL)
    {
        if (strcmp(dp->d_name, ".") != 0 && strcmp(dp->d_name, "..") != 0)
        {
            strcpy(path, basePath);
            strcat(path, "/");
            strcat(path, dp->d_name);
            // printf("%s\n", path);
			FILE* fp = fopen(path, "r");
			if (fp == NULL) { 
				return 0; 
			} 
			fseek(fp, 0L, SEEK_END); 
			// calculating the size of the file 
			long int fsize = ftell(fp); 
			if (dp->d_type == DT_REG && fsize > 0){ // PutObject does not support empty files
				if (PutObject(c, (void*) path, strlen(path), path) < 0 ){
					printf("PutObject - %s failed.\n", path);
					return -1;
				}
			}
            if (putRecursive(c, path) < 0) return -1;
        }
    }
    closedir(dir);
	return 0;
}

int main(int argc, char* argv[]) {
	DSSClient *c;
	int fd = -1;
	int ret = -1;
	int size = 1024*1024;
	unsigned char* buff;
	unsigned char* buff1;
	int cur_page = -1;
	int max_key_len = 1024;
	char* keys_list; 
	char obj_name[256];
    	char uuid[] = "12345";

	if (argc < 4){
		printf("Invalid number of arguments given\n");
		printf("Usage: %s <endpoint_url> <access_key> <secret_key> <size_in_kb(default 1MB)>\n", argv[0]);
		printf("Endpoint URL format is '<minio_host_name_or_ip>:<port>'\n");
		return -1;
	}
	if (strchr(argv[1], ':') == NULL) {
		printf("Invalid endpoint URL\n");
		return -1;
	}
	if (argc >= 5)
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

	// upload objects with a directory structure to test LIST
	if (putRecursive(c, "/etc") < 0 ){
		printf("Uploading directory failed.\n");
		goto out;
	}
	//test the LIST
	keys_list = (char*) malloc(sizeof(char) * max_key_len * GetPageSize());
	if (keys_list == NULL) {
		printf("malloc failed for allocating buffer to store the LIST results\n");
		goto out;
	}
	while (1){
		cur_page = ListObjects(c, "", "", keys_list, cur_page); // List all objects with prefix ""
		if (cur_page == FAILURE){
			printf("ListObjects failed.\n");
			goto out;
		}
		if (cur_page == END_OF_LIST) break;
		printf("page index = %d:\n[%s]\n", cur_page, keys_list);
	}
	//test the DeleteAll
	ret = DeleteAll(c, "etc", "");
	if (ret < 0) {
		printf("DeleteAll failed.\n");
		goto out;
	}
	//retest the LIST after DeleteAll
	printf("After DeleteAll:\n");
	cur_page = -1;
	while (1){
		cur_page = ListObjects(c, "", "", keys_list, cur_page); // List all objects with prefix=""
		if (cur_page == FAILURE){
			printf("ListObjects failed.\n");
			goto out;
		}
		if (cur_page == END_OF_LIST) break;
		printf("page index = %d:\n[%s]\n", cur_page, keys_list);
	}

out:
	if (fd != -1)
		close(fd);
	free(buff);
	free(buff1);
	free(keys_list);
	return ret;
}
