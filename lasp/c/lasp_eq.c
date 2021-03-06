#include "lasp_eq.h"
#include "lasp_assert.h"

typedef struct Eq {
    Sosfilterbank* fb;
    us nfilters;
    vd ampl_values;
} Eq;

Eq* Eq_create(Sosfilterbank* fb) {
    fsTRACE(15);
    assertvalidptr(fb);
    Eq* eq = a_malloc(sizeof(Eq));
    eq->fb = fb;
    eq->nfilters = Sosfilterbank_getFilterbankSize(fb);
    eq->ampl_values = vd_alloc(eq->nfilters);
    vd_set(&(eq->ampl_values), 1.0);
    feTRACE(15);
    return eq;
}
vd Eq_equalize(Eq* eq,const vd* input_data) {
    fsTRACE(15);
    assertvalidptr(eq);
    assert_vx(input_data);
    vd result = vd_alloc(input_data->n_rows);
    dmat_set(&result, 0);
    dmat filtered = Sosfilterbank_filter(eq->fb, input_data);

    for(us filter=0;filter<eq->nfilters;filter++) {
        d ampl = *getvdval(&(eq->ampl_values), filter);
        /// TODO: Replace this code with something more fast from BLAS.
        for(us sample=0;sample<filtered.n_rows;sample++) {
            d* res = getvdval(&result, sample);
            *res = *res + *getdmatval(&filtered, sample, filter) * ampl;
        }
    }
    dmat_free(&filtered);

    feTRACE(15);
    return result;
}

void Eq_setLevels(Eq* eq,const vd* levels) {
    fsTRACE(15);
    assertvalidptr(eq);
    assert_vx(levels);
    dbgassert(levels->n_rows == eq->nfilters, "Invalid levels size");
    for(us ch=0;ch<eq->nfilters;ch++){
        d level = *getvdval(levels, ch);
        *getvdval(&(eq->ampl_values), ch) = d_pow(10, level/20);
    }

    feTRACE(15);
}

us Eq_getNLevels(const Eq* eq) {
    fsTRACE(15);
    assertvalidptr(eq);
    feTRACE(15);
    return eq->nfilters;
}

void Eq_free(Eq* eq) {
    fsTRACE(15);
    assertvalidptr(eq);
    assertvalidptr(eq->fb);
    Sosfilterbank_free(eq->fb);
    vd_free(&(eq->ampl_values));
    feTRACE(15);
}
