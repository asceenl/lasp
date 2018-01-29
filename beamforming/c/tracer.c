// si_tracer.c
//
// last-edit-by: J.A. de Jong 
// 
// Description:
//
//////////////////////////////////////////////////////////////////////
#if TRACER == 1
#include <stdio.h>
#include "tracer.h"
#include "types.h"

#ifdef _REENTRANT
#include <stdatomic.h>

_Atomic(int) TRACERNAME = ATOMIC_VAR_INIT(DEFAULTTRACERLEVEL);

void setTracerLevel(int level) {
    atomic_store(&TRACERNAME,level);
}
int getTracerLevel() {
    return atomic_load(&TRACERNAME);
}
#else

int TRACERNAME;

/* setTracerLevel and getTracerLevel are defined as macros in
 * tracer.h */
#endif



void trace_impl(const char* file,int pos, const char * string){
    printf(annestr(TRACERNAME) ":%s:%i: %s\n",file,pos,string);
}
void fstrace_impl(const char* file,int pos,const char* fn){
    printf(annestr(TRACERNAME) ":%s:%i: start function: %s()\n",file,pos,fn);
}
void fetrace_impl(const char* file,int pos,const char* fn){
    printf(annestr(TRACERNAME) ":%s:%i: end   function: %s()\n",file,pos,fn);
}
void ivartrace_impl(const char* pos,int line,const char* varname, int var){
    printf(annestr(TRACERNAME) ":%s:%i: %s = %i\n",pos,line,varname,var);
}
void uvartrace_impl(const char* pos,int line,const char* varname,size_t var){
    printf(annestr(TRACERNAME) ":%s:%i: %s = %zu\n",pos,line,varname,var);
}
void dvartrace_impl(const char* pos,int line,const char* varname, d var){
    printf(annestr(TRACERNAME) ":%s:%i: %s = %0.5e\n",pos,line,varname,var);
}
void cvartrace_impl(const char* pos,int line,const char* varname, c var){
    printf(annestr(TRACERNAME) ":%s:%i: %s = %0.5e+%0.5ei\n",pos,line,varname,creal(var),cimag(var));
}
#endif

//////////////////////////////////////////////////////////////////////
