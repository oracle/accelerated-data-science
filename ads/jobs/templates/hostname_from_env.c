// -*- coding: utf-8; -*-

// Copyright (c) 2023 Oracle and/or its affiliates.
// Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.

// Compile the file and save it into the lib directory:
// $ gcc -fPIC -shared -Wl,-soname,libhostname.so.1 -ldl -o ${CONDA_PREFIX}/lib/libhostname.so.1 hostname_from_env.c
// Run the script with two environment variables:
// $ LD_PRELOAD=${CONDA_PREFIX}/lib/libhostname.so.1 OCI__HOSTNAME=<> python script.py
#define _GNU_SOURCE
#include <dlfcn.h>
#include <netdb.h>
#include <string.h>
#include <stdlib.h>

/** @brief Returns a hostname specified as the value of environment variable.
 *
 *  The parameters are the same as the original gethostname function.
 *  See: https://www.gnu.org/software/libc/manual/html_node/Host-Identification.html#index-gethostname
 *
 *  @param name The hostname to be returned. This will contain the value from environment variable OCI__HOSTNAME
 *  @param size The length of the string s.
 *  @return The return value is 0 on success and -1 if the environment variable is not set.
 */
int gethostname(char *name, size_t size)
{
    if (getenv("OCI__HOSTNAME"))
    {
        strncpy(name, getenv("OCI__HOSTNAME"), size);
        return 0;
    }
    return -1;
}

/** @brief A wrapper of gethostbyaddr_r, to return the hostname based on the environment variable.
 *
 *  The parameters are the same as the original gethostbyaddr_r function.
 *  See: https://www.gnu.org/software/libc/manual/html_node/Host-Names.html#index-gethostbyaddr_005fr
 *
 *  When addr is the same as the value of environment variable OCI__HOSTNAME,
 *  The returned value ret->h_name will have the same value as environment variable OCI__HOSTNAME.
 */
int gethostbyaddr_r(const void *addr, socklen_t len, int type,
                    struct hostent *ret, char *buf, size_t buflen,
                    struct hostent **result, int *h_errnop)
{
    int (*originalFunc)(const void *, socklen_t, int, struct hostent *, char *, size_t, struct hostent **, int *) = dlsym(RTLD_NEXT, "gethostbyaddr_r");
    int originalReturn = originalFunc(addr, len, type, ret, buf, buflen, result, h_errnop);

    if (strcmp(getenv("OCI__HOSTNAME"), (char *)addr))
    {
        ret->h_name = getenv("OCI__HOSTNAME");
    }
    return originalReturn;
}
