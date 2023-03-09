#ifndef DSS_H
#define DSS_H

#ifdef __cplusplus
extern "C" {
#endif
typedef void* DSSClient;
DSSClient DSSClientInit(char *ip, char* user, char* passwd, char* uuid, int endpoints_per_cluster);
int GetObjectBuffer(DSSClient c, void* key, int key_len, unsigned char* buffer, long int buffer_size);
int GetObject(DSSClient c, void* key, int key_len, char* dst_file);
int PutObjectBuffer(DSSClient c, void* key, int key_len, unsigned char* buffer, long int content_length);
int DeleteObject(DSSClient c, void* key, int key_len);

#ifdef __cplusplus
}
#endif
#endif
