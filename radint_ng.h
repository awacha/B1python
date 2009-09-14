
typedef struct {
    double *q;
    double *Intensity;
    double *Error;
    double *Area;
    double *Weight;
    unsigned long N;
} RadintResult;

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
                      unsigned long Nq);

void freeRadintResult(RadintResult *rr);
