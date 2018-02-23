// lasp_assert.c
//
// Author: J.A. de Jong -ASCEE
// 
// Description:
//
//////////////////////////////////////////////////////////////////////
#include "lasp_assert.h"

#ifdef LASP_DEBUG
#include <stdlib.h>
#include <stdio.h>
#include "lasp_types.h"
#define MAX_MSG 200

void DBG_AssertFailedExtImplementation(const char* filename,
                                       us linenr,
                                       const char* extendedinfo)
{
    char scratchpad[MAX_MSG];

    sprintf(scratchpad,"ASSERT: file %s line %lu: (%s)\n",
            filename, linenr, extendedinfo);
    printf("%s\n", scratchpad);
    abort();
}

#endif
