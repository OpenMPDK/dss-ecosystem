#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dss.h"

using namespace dss;
namespace py = pybind11;

PYBIND11_MODULE(dss, m) {
	m.def("createClient", &Client::CreateClient);

    py::class_<Client>(m, "Client")
        .def(py::init<const std::string&, const std::string&, const std::string&>())
        .def("putObject", &Client::PutObject,	"Upload object to dss cluster")
        .def("getObject", &Client::GetObject,	"Download object from dss cluster")
        .def("deleteObject", &Client::DeleteObject, "Delete object from dss cluster")
        .def("listObjects", &Client::ListObjects, "List object keys with prefix", py::arg("prefix") = "")
        //.def("createBucket", &Client::CreateBucket)
        //.def("deleteBucket", &Client::DeleteBucket)
        //.def("listBuckets", &Client::ListBuckets)
		.def("getObjects", &Client::GetObjects);

	class NoIterator : std::exception {
	public:
    	const char* what() const noexcept {return "No iterator\n";}
	};

    static py::exception<NoSuchResouceError> NoSuchResourceExc(m, "NoSuchResouceError");
    static py::exception<NetworkError> NetworkExc(m, "NetworkError");
    static py::exception<GenericError> GenericExc(m, "GenericError");
    static py::exception<NoIterator> LastIterExc(m, "NoIterator");

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const NoSuchResouceError &e) {
            NoSuchResourceExc(e.what());
        } catch (const NetworkError &e) {
            NetworkExc(e.what());
        } catch (const GenericError &e) {
            GenericExc(e.what());
        } catch (const NoIterator &e) {
            LastIterExc(e.what());
        }
    });

    py::class_<Objects>(m, "Objects")
        .def("__iter__", [](Objects &objs) {
        	if (objs.GetObjKeys() < 0)
        		throw NoIterator();
        	return py::make_iterator(objs.begin(), objs.end());
        }, py::keep_alive<0, 1>());

    //m.def("init", &InitAwsAPI);
    //m.def("fini", &FiniAwsAPI);
}

