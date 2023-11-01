#ifndef DSS_H
#define DSS_H

#ifdef __cplusplus
extern "C" {
#endif
typedef void* DSSClient;
DSSClient DSSClientInit(char *ip, char* user, char* passwd, char* uuid, int endpoints_per_cluster);
int GetObjectBuffer(DSSClient c, void* key, int key_len, unsigned char* buffer, long int buffer_size);
int PutObjectBuffer(DSSClient c, void* key, int key_len, unsigned char* buffer, long int content_length);
int GetObject(DSSClient c, void* key, int key_len, char* dst_file);
int PutObject(DSSClient c, void* key, int key_len, char* src_file);
int DeleteObject(DSSClient c, void* key, int key_len);
char* ListObjects(DSSClient c, char* prefix, char* delimit);

#ifdef __cplusplus
}
#endif
#endif
