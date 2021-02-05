/******************************************************
 *
 *
 *  Samsung copyright 2021
 *
 *
 *****************************************************/
#ifndef __INSDB_INC_PR_H
#define __INSDB_INC_PR_H

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
        	printf("%s\n", file);
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
