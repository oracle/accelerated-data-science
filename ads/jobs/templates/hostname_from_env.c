// Compile the file and save it into the lib directory:
// gcc -fPIC -shared -Wl,-soname,libhostname.so.1 -ldl -o ${CONDA_PREFIX}/lib/libhostname.so.1 hostname_from_env.c
// Run the script with two environment variables:
// LD_PRELOAD=conda_prefix/lib/libhostname.so.1 OCI__HOSTNAME=<> python script.py

#include <string.h>
#include <stdlib.h>

/** @brief Returns a hostname specified as the value of environment variable.
 *
 *  The parameters are the same as the original gethostname function.
 *  See: https://www.gnu.org/software/libc/manual/html_node/Host-Identification.html#index-gethostname
 *
 *  @param name The hostname to be returned.
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
