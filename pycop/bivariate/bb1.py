
def bivariate_BB1_cdf(u, v, teta, delta):
    return (1+( (u**(-teta) -1)**delta +  (v**(-teta) -1)**delta )**(1/delta) )**(-1/teta)

def bivariate_BB1_pdf(u, v, teta, delta):
    x = (u**(-teta)-1)**(delta)
    y = (v**(-teta)-1)**(delta)
    return ((1+(x+y)**(1/delta))**(-1/teta-2))*((x+y)**(1/delta-2))*(teta*(delta-1)+(teta*delta+1)*(x+y)**(1/delta))*((x*y)**(1-(1/delta)))*((u*v)**(-teta-1))
