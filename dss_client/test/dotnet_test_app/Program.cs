using System;
 
using System.Runtime.InteropServices;

using System.Diagnostics;

using System.Security.Cryptography; // for System.Security.Cryptography.MD5

unsafe public class Tester
{
        public const string dss_lib = "libdss.so";
        [DllImport(dss_lib, EntryPoint="DSSClientInit")]
        unsafe static extern void* DSSClientInit(string end_point, string access_key, string secret_key, string uuid, int name_len);
        [DllImport(dss_lib, EntryPoint="PutObjectBuffer")]
        unsafe static extern int PutObjectBuffer(void* dss_client, string key, int key_len, byte[] buf, long buf_size);

        [DllImport(dss_lib, EntryPoint="GetObjectBuffer")]
        unsafe static extern int GetObjectBuffer(void* dss_client, string key, int key_len, byte* buf, long buf_size);

        [DllImport(dss_lib, EntryPoint="GetObject")]
        static extern int GetObject(void* dss_client, string key, int key_len, string dest_fn);
        [DllImport(dss_lib, EntryPoint="PutObject")]
        static extern int PutObject(void* dss_client, string key, int key_len, string src_fn);

        [DllImport(dss_lib, EntryPoint="DeleteObject")]
        static extern int DeleteObject(void* dss_client, string key, int key_len);
        [DllImport(dss_lib, EntryPoint="ListObjects")]
        unsafe static extern IntPtr ListObjects(void* dss_client, string prefix, string delimit);
        static public string CalculateMD5(string filename)
        {
                using (var md5 = MD5.Create())
                {
                        using (var stream = File.OpenRead(filename))
                        {
                                var hash = md5.ComputeHash(stream);
                                return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
                                // return BitConverter.ToString(hash);
                        }
                }
        }

        static void DisplayKeys(IntPtr char_ptr){
                string managedString = Marshal.PtrToStringAnsi(char_ptr);
                string[] keys = managedString.Remove(managedString.Length-1,1).Split('\n');
                foreach (var key in keys)
                {
                        Console.WriteLine($"obj: {key}");
                }
        }

        unsafe static public void Main(string[] args)
        {
                if (args.Length < 3){
                        Console.WriteLine("Please use valid args: <endpoint_url> <access_key> <secret_key>");
                        return;
                }
                string end_point = args[0];
                string name= args[1];
                string psw = args[2];
                string obj_key0 = "exp0";
                string obj_key1 = "exp1";
                string get_res0_fn = "get1.result";
                string get_res1_fn = "get2.result";
                int size = 1024*1024; // 1024 KB 1MB
                string fname = "path_of_your_local_file";
                Console.WriteLine("---------------C#: test the creation of dss client -------------");
                void* dss_client = DSSClientInit(end_point, name, psw, "12345", 256);
                if (dss_client == null) {Console.WriteLine("Client creation Failed"); return;}
                Console.WriteLine("---------------creation of dss client: PASSED -------------");
                Console.WriteLine("---------------C#: test GetObjectBuffer/PutObjectBuffer -------------");
                byte[] input_of_put = File.ReadAllBytes(fname);
                if (PutObjectBuffer(dss_client, obj_key0, obj_key0.Length, input_of_put, size) < 0){
                        Console.WriteLine("PutObjectBuffer Failedfor {}", obj_key0); return;}
                byte* output_of_get = stackalloc byte[size];
                if (GetObjectBuffer(dss_client, obj_key0, obj_key0.Length, output_of_get, size) < 0){
                        Console.WriteLine("GetObjectBuffer Failed for {}", obj_key0); return;}
                for (int i=0;i < size; i++){
                        Debug.Assert(input_of_put[i]==output_of_get[i], "GET/PUT result mismatch");
                }
                Console.WriteLine("---------------C#: GetObjectBuffer/PutObjectBuffer: PASSED -------------");
                Console.WriteLine("---------------C#: GetObject/PutObject -------------");
                int ret_get0 = GetObject(dss_client, obj_key0, obj_key0.Length, get_res0_fn);
                string s0 = CalculateMD5(get_res0_fn);
                Console.WriteLine("{0}: {1}", get_res0_fn, s0);
                PutObject(dss_client,  obj_key1, obj_key1.Length, get_res0_fn);
                int ret_get1 = GetObject(dss_client, obj_key1, obj_key1.Length, get_res1_fn);
                string s1 = CalculateMD5(get_res1_fn);
                Console.WriteLine("{0}: {1}", get_res1_fn, s1);
                Debug.Assert(s0.Equals(s1), "GET/PUT result mismatch");
                Console.WriteLine("---------------C#: GetObject/PutObject: PASSED -------------");
                Console.WriteLine("---------------C#: DeleteObject/ListObjects -------------");
                byte* buf_before_delete = stackalloc byte[size];
                if (GetObjectBuffer(dss_client, obj_key0, obj_key0.Length, buf_before_delete, size) < 0){
                        Console.WriteLine("GetObjectBuffer Failed for {0}", obj_key0); return;
                }
                Debug.Assert(buf_before_delete!=null, string.Format("failed to get the object {0}, choose another key to delete", obj_key0 ) );
                Console.WriteLine("before delete");
                IntPtr charPtr = ListObjects(dss_client, "ex", "/");
                DisplayKeys(charPtr);
                Console.WriteLine("after delete");
                DeleteObject(dss_client, obj_key0, obj_key0.Length);
                IntPtr charPtr2 = ListObjects(dss_client, "e", "/");
                DisplayKeys(charPtr2);
                byte* buf_unreachable = stackalloc byte[size];
                if (GetObjectBuffer(dss_client, obj_key0, obj_key0.Length, buf_unreachable, size) < 0){ // expected to fail: "Exception caught in GetObjectBuffer for exp0"
                        Console.WriteLine("GetObjectBuffer Failed for {0}", obj_key0);
                }
                Marshal.FreeCoTaskMem(charPtr);
                Marshal.FreeCoTaskMem(charPtr2);
                Console.WriteLine("---------------C#: DeleteObject/ListObjects: PASSED -------------");
        }
}
