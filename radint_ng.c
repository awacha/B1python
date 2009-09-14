#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define DEBUG
#define WEIGHTBYERROR
#define WEIGHTBYQDQ2

#define HC 12398.419 /*eV*Angstrom*/

typedef struct {
    double *q;
    double *Intensity;
    double *Error;
    double *Area;
    double *Weight;
    unsigned long N;
} RadintResult;


void freeRadintResult(RadintResult *rr)
{
    if (rr)
        {
            free(rr->q); free(rr->Intensity); free(rr->Error); free(rr->Area);
            free(rr->Weight);
            free(rr);
        }
}

RadintResult *doradint(double *data,
                      double *error,
                      unsigned short *mask,
                      unsigned long Nrow,
                      unsigned long Ncol,
                      double energy,
                      double distance,
                      double xresol,
                      double yresol,
                      double bcx,
                      double bcy,
                      double *q,
                      unsigned long Nq)
{
    unsigned long x,y;
    RadintResult *res;
    
    if ((data==NULL) || (error==NULL) || (mask==NULL))
        {
#ifdef DEBUG
            fprintf(stderr,"radint_ng.c: data or error or mask was NULL\n");
#endif
            return NULL;
        }
    if (q!=NULL)
        { /*make a copy of q*/
            double *q2;
            q2=(double*)malloc(sizeof(double)*Nq);
            for (x=0; x<Nq; x++)
                q2[x]=q[x];
            q=q2;
        }
    if (q==NULL)
        {
            double dmax=0;
            double dmin=(Nrow+Ncol)*1000;
            double qmin=4*M_PI*sin(0.5*atan(dmin/distance))*energy/HC;
            double qmax=4*M_PI*sin(0.5*atan(dmax/distance))*energy/HC;
            double r;
            for (x=0; x<Nrow; x++)
                for (y=0; y<Ncol; y++)
                    if (!mask[y*Nrow+x])
                        {
                            double qval;
                            r=sqrt((x-(bcx-1))*(x-(bcx-1))+(y-(bcy-1))*(y-(bcy-1)));
                            qval=4*M_PI*sin(0.5*atan(sqrt(xresol*xresol*(x-(bcx-1))*(x-(bcx-1))+
                                                          yresol*yresol*(y-(bcy-1))*(y-(bcy-1)))/distance))*energy/HC;
                            if (r<dmin) dmin=r;
                            if (r>dmax) dmax=r;
                            if (qval<qmin) qmin=qval;
                            if (qval>qmax) qmax=qval;
                        }
            if (dmax<=dmin)  /*not enough non-masked points*/
                return NULL;
            if (Nq<=0)
                Nq=ceil(dmax-dmin)*3;
            q=(double *)malloc(Nq*sizeof(double));
            for (x=0; x<Nq; x++)
                q[x]=qmin+(double)((qmax-qmin)/Nq*x);
        }
    /*we now have q, which is _ours_.*/
    res=(RadintResult *)malloc(sizeof(RadintResult));
    if (!res)
        {
            free(q);
#ifdef DEBUG
            fprintf(stderr,"radint_ng.c: cannot allocate res\n");
#endif
            return NULL;
        }
    res->Intensity=(double*)calloc(Nq,sizeof(double));
    if (!(res->Intensity))
        {
#ifdef DEBUG
            fprintf(stderr,"radint_ng.c: cannot allocate res->Intensity\n");
#endif
            free(q); free(res);
            return NULL;
        }
    res->Error=(double*)calloc(Nq,sizeof(double));
    if (!(res->Error))
        {
#ifdef DEBUG
            fprintf(stderr,"radint_ng.c: cannot allocate res->Error\n");
#endif
            free(q); free(res->Intensity); free(res);
            return NULL;
        }
    res->Area=(double*)calloc(Nq,sizeof(double));
    if (!(res->Area))
        {
#ifdef DEBUG
            fprintf(stderr,"radint_ng.c: cannot allocate res->Area\n");
#endif
            free(q); free(res->Intensity); free(res->Error); free(res);
            return NULL;
        }
    res->Weight=(double*)calloc(Nq,sizeof(double));
    if (!(res->Weight))
        {
#ifdef DEBUG
            fprintf(stderr,"radint_ng.c: cannot allocate res->Weight\n");
#endif
            free(q); free(res->Intensity); free(res->Error); free(res->Area); free(res);
            return NULL;
        }
    res->N=Nq;
    res->q=q;
    for(x=0; x<Nrow; x++)
        for(y=0; y<Ncol; y++)
            {
                double rval, qval, qdq2val, tg2theta;
                double curweight,lowborder;
                unsigned long qind;
                if (mask[y*Nrow+x]!=0)
                    continue;
                rval=sqrt(xresol*xresol*(x-(bcx-1))*(x-(bcx-1))+yresol*yresol*(y-(bcy-1))*(y-(bcy-1)));
                qval=4*M_PI*sin(0.5*atan(rval/distance))*energy/HC;
                tg2theta=rval/distance;
                qdq2val=qval*pow(2*M_PI*energy/(HC*distance),2)*(2+pow(tg2theta,2)+2*sqrt(1+tg2theta*tg2theta))/(pow(1+tg2theta*tg2theta+sqrt(1+tg2theta*tg2theta),2)*sqrt(1+tg2theta*tg2theta));
                if ((qval<q[0]) || (qval>q[Nq-1]))
                    continue;
                for (qind=0;qind<Nq; qind++)
                    {
                        lowborder=0.5*(q[qind-1]+q[qind]);
                        if (qval>=lowborder)
                            break;
                    }
                if (qind>=res->N)
                    continue;
                /*now qind contains the index of the q-bin where the current pixel belongs*/
                curweight=1;
#ifdef WEIGHTBYERROR
                curweight=curweight/(error[y*Nrow+x]*error[y*Nrow+x]);
#endif
#ifdef WEIGHTBYQDQ2
                curweight=curweight*qdq2val;
#endif                
                res->Weight[qind]+=curweight;
                res->Intensity[qind]+=data[y*Nrow+x]*curweight;
                res->Area[qind]+=1;
            }
    for(x=0; x<res->N; x++)
        {
            if (res->Weight[x]!=0)
                {
                    res->Intensity[x]/=res->Weight[x];                    
                    res->Error[x]=1/res->Weight[x];
                }
        }
#ifdef DEBUG
    fprintf(stderr,"radint_ng.c: now returning with results.\n");
#endif
    return res;
}