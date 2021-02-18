#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dss.h"

using namespace dss;
namespace py = pybind11;

PYBIND11_MODULE(dss, m) {
    py::class_<SesOptions>(m, "clientOption")
    	.def(py::init<>())
		.def_readwrite("scheme", &SesOptions::scheme)
		.def_readwrite("useDualStack", &SesOptions::useDualStack)
		.def_readwrite("maxConnections", &SesOptions::maxConnections)
		.def_readwrite("httpRequestTimeoutMs", &SesOptions::httpRequestTimeoutMs)
		.def_readwrite("requestTimeoutMs", &SesOptions::requestTimeoutMs)
		.def_readwrite("connectTimeoutMs", &SesOptions::connectTimeoutMs)
		.def_readwrite("enableTcpKeepAlive", &SesOptions::enableTcpKeepAlive)
		.def_readwrite("tcpKeepAliveIntervalMs", &SesOptions::tcpKeepAliveIntervalMs);
#if 0
	m.def("createClient", static_cast<Client* (Client::*)(const std::string&, const std::string&, const std::string&)>(&Client::CreateClient),
		py::arg("url"),
		py::arg("username"),
		py::arg("password"));
#endif
	m.def("createClient", &Client::CreateClient,
		py::arg("url"),
		py::arg("username"),
		py::arg("password"),
		py::arg("options") = SesOptions());
    
    py::class_<Client>(m, "Client")
        .def("putObject", &Client::PutObject,	"Upload object to dss cluster",
        	py::arg("key"),
        	py::arg("file_path"))
        .def("getObject", &Client::GetObject,	"Download object from dss cluster",
        	py::arg("key"),
        	py::arg("file_path"))
        .def("deleteObject", &Client::DeleteObject, "Delete object from dss cluster",
        	py::arg("key"))
        .def("listObjects", &Client::ListObjects, "List object keys with prefix",
        	 py::arg("prefix") = "")
		.def("getObjects", &Client::GetObjects, "Create a iterable key list",
			py::arg("prefix") = "",
			py::arg("limit") = DSS_PAGINATION_DEFAULT);

	class NoIterator : std::exception {
	public:
    	const char* what() const noexcept {return "No iterator\n";}
	};

    static py::exception<NoSuchResourceError> NoSuchResourceExc(m, "NoSuchResouceError");
 	static py::exception<DiscoverError> DiscoverExc(m, "DiscoverError");
    static py::exception<NetworkError> NetworkExc(m, "NetworkError");
    static py::exception<GenericError> GenericExc(m, "GenericError");
    static py::exception<NoIterator> LastIterExc(m, "NoIterator");

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const NoSuchResourceError &e) {
            NoSuchResourceExc(e.what());
        } catch (const DiscoverError &e) {
            DiscoverExc(e.what());
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
}

