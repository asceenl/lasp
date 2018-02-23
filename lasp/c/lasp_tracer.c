// lasp_tracer.c
//
// last-edit-by: J.A. de Jong 
// 
// Description:
//
//////////////////////////////////////////////////////////////////////
#if TRACER == 1
#include <stdio.h>
#include "lasp_tracer.h"
#include "lasp_types.h"

#ifdef _REENTRANT
#include <stdatomic.h>
static _Thread_local us ASCEE_FN_LEVEL = 0;

static _Atomic(int) TRACERNAME = ATOMIC_VAR_INIT(DEFAULTTRACERLEVEL);


void setTracerLevel(int level) {
    atomic_store(&TRACERNAME,level);
}
int getTracerLevel() {
    return atomic_load(&TRACERNAME);
}

#else

int TRACERNAME;
static us ASCEE_FN_LEVEL = 0;
/* setTracerLevel and getTracerLevel are defined as macros in
 * tracer.h */
#endif

void indent_trace() {
    for(us i=0;i<ASCEE_FN_LEVEL;i++) {
        printf("--");
    }
    printf("* ");
}



void trace_impl(const char* file,int pos, const char * string){
    indent_trace();
    printf(annestr(TRACERNAME) ":%s:%i: %s\n",file,pos,string);
}
void fstrace_impl(const char* file,int pos,const char* fn){
    ASCEE_FN_LEVEL++;
    indent_trace();
    printf(annestr(TRACERNAME) ":%s:%i: start function: %s()\n",file,pos,fn);
}
void fetrace_impl(const char* file,int pos,const char* fn){
    indent_trace();
    printf(annestr(TRACERNAME) ":%s:%i: end   function: %s()\n",file,pos,fn);
    ASCEE_FN_LEVEL--;
}
void ivartrace_impl(const char* pos,int line,const char* varname, int var){
    indent_trace();
    printf(annestr(TRACERNAME) ":%s:%i: %s = %i\n",pos,line,varname,var);
}
void uvartrace_impl(const char* pos,int line,const char* varname,size_t var){
    indent_trace();
    printf(annestr(TRACERNAME) ":%s:%i: %s = %zu\n",pos,line,varname,var);
}
void dvartrace_impl(const char* pos,int line,const char* varname, d var){
    indent_trace();
    printf(annestr(TRACERNAME) ":%s:%i: %s = %0.5e\n",pos,line,varname,var);
}
void cvartrace_impl(const char* pos,int line,const char* varname, c var){
    indent_trace();
    printf(annestr(TRACERNAME) ":%s:%i: %s = %0.5e+%0.5ei\n",pos,line,varname,creal(var),cimag(var));
}
#endif

//////////////////////////////////////////////////////////////////////
