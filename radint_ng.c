#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define DEBUG
#define WEIGHTBYERROR
#define WEIGHTBYQDQ2

#define HC 12398.419 /*eV*Angstrom*/


int doradint(double *data, double *error, unsigned short *mask,
            unsigned long Nrow, unsigned long Ncol,
            double energy, double distance, double xresol, double yresol,
            double bcx, double bcy, 
            double *q, double *Intensity, double *Error, double *Area, double *Weight,
            unsigned long Nq)
{
    unsigned long x,y;
    union {
            double *ptr;
            unsigned char chr[4];
        } tmp;
    tmp.ptr=q;
    fprintf(stderr,"q pointer: %02x %02x %02x %02x\n",tmp.chr[0],tmp.chr[1],tmp.chr[2],tmp.chr[3]);
    tmp.ptr=Intensity;
    fprintf(stderr,"Intensity pointer: %02x %02x %02x %02x\n",tmp.chr[0],tmp.chr[1],tmp.chr[2],tmp.chr[3]);
    tmp.ptr=Error;
    fprintf(stderr,"Error pointer: %02x %02x %02x %02x\n",tmp.chr[0],tmp.chr[1],tmp.chr[2],tmp.chr[3]);
    tmp.ptr=Area;
    fprintf(stderr,"Area pointer: %02x %02x %02x %02x\n",tmp.chr[0],tmp.chr[1],tmp.chr[2],tmp.chr[3]);
    tmp.ptr=Weight;
    fprintf(stderr,"Weight pointer: %02x %02x %02x %02x\n",tmp.chr[0],tmp.chr[1],tmp.chr[2],tmp.chr[3]);
    
    if ((data==NULL) || (error==NULL) || (mask==NULL) || (q==NULL) || 
        (Intensity==NULL) || (Error=NULL) || (Area==NULL) || (Weight==NULL))
        {
#ifdef DEBUG
            fprintf(stderr,"radint_ng.c: data or error or mask or q or intensity or error or area or weight was NULL\n");
#endif
            return 0;
        }
#ifdef DEBUG
    fprintf(stderr,"Before integrating\n");
#endif
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
                if (qind>=Nq)
                    continue;
                /*now qind contains the index of the q-bin where the current pixel belongs*/
                curweight=1;
#ifdef WEIGHTBYERROR
                curweight=curweight/(error[y*Nrow+x]*error[y*Nrow+x]);
#endif
#ifdef WEIGHTBYQDQ2
                curweight=curweight*qdq2val;
#endif                
                Weight[qind]+=curweight;
                Intensity[qind]+=data[y*Nrow+x]*curweight;
                Area[qind]+=1;
            }
#ifdef DEBUG
    fprintf(stderr,"Integration succeeded. Normalizing...\n");
#endif
    for(x=0; x<Nq; x++)
        {
#ifdef DEBUG
            fprintf(stderr,"Normalizing element %lu...\n",x);
#endif
            if (Weight[x]!=0)
                {
                    fprintf(stderr,"Weight is not zero\n");
                    Intensity[x]=Intensity[x]/=Weight[x];  
                    fprintf(stderr,"Calculating error\n");
                    Error[x]=1.0/Weight[x];
                    fprintf(stderr,"Error has been calculated\n");
                }
        }
#ifdef DEBUG
    fprintf(stderr,"radint_ng.c: now returning with results.\n");
#endif
    return 1;
}