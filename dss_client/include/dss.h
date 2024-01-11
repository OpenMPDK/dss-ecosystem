#ifndef DSS_H
#define DSS_H

#define FAILURE -1
#define END_OF_LIST -2

#ifdef __cplusplus
extern "C" {
#endif
typedef void* DSSClient;
DSSClient DSSClientInit(char *ip, char* user, char* passwd, char* uuid, int endpoints_per_cluster);
int GetObjectBuffer(DSSClient c, void* key, int key_len, unsigned char* buffer, long int buffer_size);
int GetObject(DSSClient c, void* key, int key_len, char* dst_file);
int PutObject(DSSClient c, void* key, int key_len, char* src_file);
int PutObjectBuffer(DSSClient c, void* key, int key_len, unsigned char* buffer, long int content_length);
int DeleteObject(DSSClient c, void* key, int key_len);
int ListObjects(DSSClient c, char* prefix, char* delimit, char* obj_keys, int cur_pg);
int DeleteAll(DSSClient c, char* prefix, char* delimit);
int GetPageSize();
#ifdef __cplusplus
}
#endif
#endif
