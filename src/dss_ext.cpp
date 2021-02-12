#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dss.h"

using namespace dss;
namespace py = pybind11;

PYBIND11_MODULE(dss, m) {
	m.def("createClient", &Client::CreateClient,
		py::arg("url"),
		py::arg("username"),
		py::arg("password"));

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

    py::class_<Objects>(m, "Objects")
        .def("__iter__", [](Objects &objs) {
        	if (objs.GetObjKeys() < 0)
        		throw NoIterator();
        	return py::make_iterator(objs.begin(), objs.end());
        }, py::keep_alive<0, 1>());

    static py::exception<NoSuchResouceError> NoSuchResourceExc(m, "NoSuchResouceError");
 	static py::exception<DiscoverError> DiscoverExc(m, "DiscoverError");
    static py::exception<NetworkError> NetworkExc(m, "NetworkError");
    static py::exception<GenericError> GenericExc(m, "GenericError");
    static py::exception<NoIterator> LastIterExc(m, "NoIterator");

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const NoSuchResouceError &e) {
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
}

