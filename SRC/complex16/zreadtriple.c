/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief
 *
 */
#include <stdio.h>
#include "superlu_zdefs.h"

#undef EXPAND_SYM

/*! brief
 *
 * <pre>
 * Output parameters
 * =================
 *   (nzval, rowind, colptr): (*rowind)[*] contains the row subscripts of
 *      nonzeros in columns of matrix A; (*nzval)[*] the numerical values;
 *	column i of A is given by (*nzval)[k], k = (*rowind)[i],...,
 *      (*rowind)[i+1]-1.
 * </pre>
 */

void
zreadtriple_dist(FILE *fp, int_t *m, int_t *n, int_t *nonz,
	    doublecomplex **nzval, int_t **rowind, int_t **colptr)
{
    int_t    j, k, jsize, nnz, nz, new_nonz;
    doublecomplex *a, *val;
    int_t    *asub, *xa, *row, *col;
    int_t    zero_base = 0;

    /* 	File format:
     *    First line:  #rows    #non-zero
     *    Triplet in the rest of lines:
     *                 row    col    value
     */

#ifdef _LONGINT
    fscanf(fp, "%lld%lld%lld", m, n, nonz);
#else
    fscanf(fp, "%d%d%d", m, n, nonz);
#endif

#ifdef EXPAND_SYM
    new_nonz = 2 * *nonz - *n;
#else
    new_nonz = *nonz;
#endif
    *m = *n;
    printf("m %lld, n %lld, nonz %lld\n", (long long) *m, (long long) *n, (long long) *nonz);
    zallocateA_dist(*n, new_nonz, nzval, rowind, colptr); /* Allocate storage */
    a    = *nzval;
    asub = *rowind;
    xa   = *colptr;

    if ( !(val = (doublecomplex *) SUPERLU_MALLOC(new_nonz * sizeof(doublecomplex))) )
        ABORT("Malloc fails for val[]");
    if ( !(row = (int_t *) SUPERLU_MALLOC(new_nonz * sizeof(int_t))) )
        ABORT("Malloc fails for row[]");
    if ( !(col = (int_t *) SUPERLU_MALLOC(new_nonz * sizeof(int_t))) )
        ABORT("Malloc fails for col[]");

    for (j = 0; j < *n; ++j) xa[j] = 0;

    /* Read into the triplet array from a file */
    for (nnz = 0, nz = 0; nnz < *nonz; ++nnz) {

#ifdef _LONGINT
        fscanf(fp, "%lld%lld%lf%lf\n", &row[nz], &col[nz], &val[nz].r, &val[nz].i);
#else // int
        fscanf(fp, "%d%d%lf%lf\n", &row[nz], &col[nz], &val[nz].r, &val[nz].i);
#endif

	if ( nnz == 0 ) { /* first nonzero */
	    if ( row[0] == 0 || col[0] == 0 ) {
		zero_base = 1;
		printf("triplet file: row/col indices are zero-based.\n");
	    } else {
		printf("triplet file: row/col indices are one-based.\n");
     	    }
        }

	if ( !zero_base ) {
	    /* Change to 0-based indexing. */
	    --row[nz];
	    --col[nz];
	}

	if (row[nz] < 0 || row[nz] >= *m || col[nz] < 0 || col[nz] >= *n
	    /*|| val[nz] == 0.*/) {
	    fprintf(stderr, "nz " IFMT ", (" IFMT ", " IFMT ") = {%e\t%e} out of bound, removed\n",
		    nz, row[nz], col[nz], val[nz].r, val[nz].i);
	    exit(-1);
	} else {
	    ++xa[col[nz]];
#ifdef EXPAND_SYM
	    if ( row[nz] != col[nz] ) { /* Excluding diagonal */
	      ++nz;
	      row[nz] = col[nz-1];
	      col[nz] = row[nz-1];
	      val[nz] = val[nz-1];
	      ++xa[col[nz]];
	    }
#endif
	    ++nz;
	}
    }

    *nonz = nz;
#ifdef EXPAND_SYM
    printf("new_nonz after symmetric expansion:\t%d\n", *nonz);
#endif


    /* Initialize the array of column pointers */
    k = 0;
    jsize = xa[0];
    xa[0] = 0;
    for (j = 1; j < *n; ++j) {
	k += jsize;
	jsize = xa[j];
	xa[j] = k;
    }

    /* Copy the triplets into the column oriented storage */
    for (nz = 0; nz < *nonz; ++nz) {
	j = col[nz];
	k = xa[j];
	asub[k] = row[nz];
	a[k] = val[nz];
	++xa[j];
    }

    /* Reset the column pointers to the beginning of each column */
    for (j = *n; j > 0; --j)
	xa[j] = xa[j-1];
    xa[0] = 0;

    SUPERLU_FREE(val);
    SUPERLU_FREE(row);
    SUPERLU_FREE(col);

#ifdef CHK_INPUT
    int i;
    for (i = 0; i < *n; i++) {
	printf("Col %d, xa %d\n", i, xa[i]);
	for (k = xa[i]; k < xa[i+1]; k++)
	    printf("%d\t%16.10f\n", asub[k], a[k]);
    }
#endif

}


void zreadrhs(int m, doublecomplex *b)
{
    FILE *fp;
    int i;

    if ( !(fp = fopen("b.dat", "r")) ) {
        fprintf(stderr, "zreadrhs: file does not exist\n");
	exit(-1);
    }
    for (i = 0; i < m; ++i)
      fscanf(fp, "%lf%lf\n", &(b[i].r), &(b[i].i));
    /*        readpair_(j, &b[i]);*/

    fclose(fp);
}


