// ascee_tracer.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
// Basic tracing code for debugging.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef ASCEE_TRACER_H
#define ASCEE_TRACER_H
#include "lasp_types.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static inline void clearScreen() {
    printf("\033c\n");
}

/** 
 * Indent the rule for tracing visibility.
 */
void indent_trace();

// Some console colors
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

// Not so interesting part
#define rawstr(x) #x
#define namestr(x) rawstr(x)
#define annestr(x) namestr(x)
#define FILEWITHOUTPATH ( strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__ )
// #define POS annestr(FILEWITHOUTPATH) ":" # __LINE__ <<  ": "
// End not so interesting part

/**
 * Produce a debug warning
 */
#define DBGWARN(a) \
    printf(RED);                                          \
    printf("%s(%d): ",                                    \
           __FILE__,                                      \
           __LINE__                                       \
           );                                             \
    printf(a);                                            \
    printf(RESET "\n");

/**
 * Produce a runtime warning
 */
#define WARN(a)              \
    printf(RED);             \
    printf("WARNING: ");     \
    printf(a);               \
    printf(RESET "\n");

/**
 * Fatal error, abort execution
 */
#define FATAL(a)         \
    WARN(a);             \
    abort(); 




// **************************************** Tracer code
#ifndef TRACERPLUS
#define TRACERPLUS (0)
#endif

// If PP variable TRACER is not defined, we automatically set it on.
#ifndef TRACER
#define TRACER 1
#endif

#if TRACER == 1
#ifndef TRACERNAME

#ifdef __GNUC__
#warning TRACERNAME name not set, sol TRACERNAME set to 'defaulttracer'
#else
#pragma message("TRACERNAME name not set, sol TRACERNAME set to defaulttracer")
#endif

#define TRACERNAME defaulttracer
#endif	// ifndef TRACERNAME

// Define this preprocessor definition to overwrite
// Use -O flag for compiler to remove the dead functions!
// In that case all cout's for TRACE() are removed from code
#ifndef DEFAULTTRACERLEVEL
#define DEFAULTTRACERLEVEL (15)
#endif

#ifdef _REENTRANT
/** 
 * Set the tracer level at runtime
 *
 * @param level 
 */
void setTracerLevel(int level);

/** 
 * Obtain the tracer level
 *
 * @return level
 */
int getTracerLevel();

#else  // Not reentrant
extern int TRACERNAME;
#define setTracerLevel(a) TRACERNAME = a;
static inline int getTracerLevel() { return TRACERNAME;}
#endif

#include "lasp_types.h"

// Use this preprocessor command to introduce one TRACERNAME integer per unit
/* Introduce one static logger */
// We trust that the compiler will eliminate 'dead code', which means
// that if variable BUILDINTRACERLEVEL is set, the inner if statement
// will not be reached.
void trace_impl(const char* pos,int line,const char * string);
void fstrace_impl(const char* file,int pos,const char* fn);
void fetrace_impl(const char* file,int pos,const char* fn);
void dvartrace_impl(const char* pos,int line,const char* varname,d var);
void cvartrace_impl(const char* pos,int line,const char* varname,c var);
void ivartrace_impl(const char* pos,int line,const char* varname,int var);
void uvartrace_impl(const char* pos,int line,const char* varname,size_t var);

/**
 * Print a trace string
 */
#define TRACE(level,trace_string)                                       \
    if (level+TRACERPLUS>=getTracerLevel())						\
		{										\
			trace_impl(FILEWITHOUTPATH,__LINE__,trace_string );	\
		}

#define SFSG TRACE(100,"SFSG")

/**
 * Print start of function string
 */
#define fsTRACE(level)                                       \
    if (level+TRACERPLUS>=getTracerLevel())						\
        {                                                               \
            fstrace_impl(FILEWITHOUTPATH,__LINE__, __FUNCTION__ ); \
        }

/**
 * Print end of function string
 */
#define feTRACE(level)                                       \
    if (level+TRACERPLUS>=getTracerLevel())						\
        {                                                               \
            fetrace_impl(FILEWITHOUTPATH,__LINE__, __FUNCTION__ ); \
        }


/**
 * Trace an int variable
 */
#define iVARTRACE(level,trace_var)				\
	if (level+TRACERPLUS>=getTracerLevel())						\
		{										\
			ivartrace_impl(FILEWITHOUTPATH,__LINE__,#trace_var,trace_var);	\
		}

/**
 * Trace an unsigned int variable
 */
#define uVARTRACE(level,trace_var)				\
	if (level+TRACERPLUS>=getTracerLevel())						\
		{										\
			uvartrace_impl(FILEWITHOUTPATH,__LINE__,#trace_var,trace_var);	\
		}
/**
 * Trace a floating point value
 */
#define dVARTRACE(level,trace_var)				\
	if (level+TRACERPLUS>=getTracerLevel())						\
		{										\
			dvartrace_impl(FILEWITHOUTPATH,__LINE__,#trace_var,trace_var);	\
		}
/**
 * Trace a complex floating point value
 */
#define cVARTRACE(level,trace_var)				\
	if (level+TRACERPLUS>=getTracerLevel())						\
		{										\
			cvartrace_impl(FILEWITHOUTPATH,__LINE__,#trace_var,trace_var);	\
		}


#else  // TRACER !=1
#define TRACE(l,a)
#define fsTRACE(l)
#define feTRACE(l)
#define setTracerLevel(a)
#define getTracerLevel()

#define iVARTRACE(level,trace_var)
#define uVARTRACE(level,trace_var)
#define dVARTRACE(level,trace_var)
#define cVARTRACE(level,trace_var)


#endif	// ######################################## TRACER ==1


#endif // ASCEE_TRACER_H
//////////////////////////////////////////////////////////////////////
