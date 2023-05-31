/**
The Clear BSD License

Copyright (c) 2023 Samsung Electronics Co., Ltd.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the disclaimer
below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of Samsung Electronics Co., Ltd. nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**/

#ifndef DSS_PR_H
#define DSS_PR_H

#include <stdint.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

enum {
	INS_LOG_UINT = 0,
	INS_LOG_ERROR,
	INS_LOG_DBG,
};

static uint32_t __LOG_LEVEL = INS_LOG_UINT;

int pr_init();

inline static int
pr_env2level(const char *f, const char *v)
{
    char *e, *p, *q;
    size_t len;

    len = strlen(v);
    e = reinterpret_cast<char*>(alloca(len + 1));
    (void) strcpy(e, v);

    for (p = strtok_r(e, ",", &q); p != NULL; p = strtok_r(NULL, ",", &q)) {
        if (strcmp(p, f) == 0)
            return INS_LOG_DBG;
    }

    return (INS_LOG_ERROR);
}

inline void
pr_timestamp(FILE *f) {
	char date[20];
	struct timeval tv;

	/* print the progname, version, and timestamp */
	gettimeofday(&tv, NULL);
	strftime(date, sizeof(date) / sizeof(*date), "%H:%M:%S", gmtime(&tv.tv_sec));
	fprintf(f, "[%s.%06ld] ", date, tv.tv_usec);
}

inline static int
_pr_err(const char *file, int line, const char *format, ...) {
    va_list ap;

	//pr_timestamp(stderr);
    fprintf(stderr, "[%s:%5d] ", file, line);

    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);

    fflush(stderr);

    return 0;
}

inline static int
_pr_debug(const char *file, int line, const char *format, ...) {
    int err = 0;
    char *s;
    va_list ap;

    if (__LOG_LEVEL == INS_LOG_UINT) {
        if ( (s = getenv("DSS_DEBUG")) ) {
            __LOG_LEVEL = pr_env2level(file, s);
        }
    }

    if (__LOG_LEVEL < INS_LOG_DBG)
        return err;

    //fprintf(stdout, "[%s:%5d] ", file, line);
	//pr_timestamp(stdout);

    va_start(ap, format);
    vfprintf(stdout, format, ap);
    va_end(ap);

    fflush(stdout);

    return err;
}

inline static void
_pr_key(const char *file, int line, const char *k, uint32_t sz) {
    char buf[sz + 1] = {0};

    if (__LOG_LEVEL < INS_LOG_DBG)
        return;

    snprintf(buf, sz+1, "%s", k);
    fprintf(stdout, "key: %s size=%u\n", buf, sz);
}

inline static int
pr_info(const char *format, ...) {
    int err = 0;
    va_list ap;

	pr_timestamp(stdout);

    va_start(ap, format);
    vfprintf(stdout, format, ap);
    va_end(ap);

    fflush(stdout);

    return err;
}

#define pr_err(fmt, args...)        \
          _pr_err(__FILE__, __LINE__, fmt, ## args)

#define pr_key(d,sz)                \
          do { if (DSS_DEBUG) _pr_key(__BASE_FILE__, __LINE__, d, sz); } while (0)

/* If DSS_DEBUG is 0, pr_debug will compile to nothing */
#define pr_debug(fmt, args...)       \
          do { if (DSS_DEBUG) _pr_debug(__BASE_FILE__, __LINE__, fmt, ## args); } while (0)

#endif
