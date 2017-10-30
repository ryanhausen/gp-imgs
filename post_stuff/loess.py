import numpy as np
def loess(x,y,dx=None):
	#dx = 0.1*(max(x)-min(x))
	if dx is None:
		dx = 1.0e-3*np.abs(max(x)-min(x))
	y_l = np.zeros(len(y),dtype=np.float32)
	w_l = np.zeros(len(y),dtype=np.float32)

	for i in range(len(x)):
		flag = 0
		dxt = dx
		#while(flag==0):
		idx = (np.abs(x-x[i])<dxt).nonzero()
		xx = x[idx]
		yy = y[idx]

			#if(len(xx)<10):
			#	dxt*=2.
			#else:
			#	flag = 1


		#wx = np.zeros(len(xx),dtype=np.float32)
		wx = (1.-(np.abs(xx-x[i])/dxt)**3)**3
		nj = len(wx)

		y_l[i] = 0.0
		if(nj>0):
			for j in range(nj):
				#if(yy[j]!=max(yy)):
					y_l[i] += wx[j]*yy[j]
					w_l[i] += wx[j]
		else:
			y_l[i] = y[i]
			w_l[i] = 1.
		y_l[i] = y_l[i]/w_l[i]
		#print(i,nj,y_l[i],min(wx),max(wx),max(yy))



	return y_l

def loessp(x,y,dx=None):
	#dx = 0.1*(max(x)-min(x))
	if dx is None:
		dx = 0.1
	y_l = np.zeros(len(y),dtype=np.float32)
	w_l = np.zeros(len(y),dtype=np.float32)

	for i in range(len(x)):
		flag = 0
		dxt = dx
		#while(flag==0):
		idx = (np.abs((x/x[i]) - 1.)<dxt).nonzero()
		xx = x[idx]
		yy = y[idx]

			#if(len(xx)<10):
			#	dxt*=2.
			#else:
			#	flag = 1


		#wx = np.zeros(len(xx),dtype=np.float32)
		wx = (1.-(np.abs( np.abs( (xx-x[i]) -1 )/dxt))**3)**3
		nj = len(wx)

		y_l[i] = 0.0
		if(nj>1):
			for j in range(nj):
				#if(yy[j]!=max(yy)):
					y_l[i] += wx[j]*yy[j]
					w_l[i] += wx[j]
		else:
			y_l[i] = y[i]
			w_l[i] = 1.
		y_l[i] = y_l[i]/w_l[i]
		#print(i,nj,y_l[i],min(wx),max(wx),max(yy))



	return y_l
def loessn(x,y,dn=None):
	#dx = 0.1*(max(x)-min(x))
	if dn is None:
		dn = 0.1*len(x)
	y_l = np.zeros(len(y),dtype=np.float32)
	w_l = np.zeros(len(y),dtype=np.float32)

	for i in range(len(x)):
		flag = 0
		#while(flag==0):
		#idx = (np.abs((x/x[i]) - 1.)<dxt).nonzero()
		idx = np.arange(i-dn,i+dn,dtype=int)
		idx = idx[( (idx>=0)&(idx<len(x))).nonzero()]
		#print(idx)
		xx = x[idx]
		yy = y[idx]

			#if(len(xx)<10):
			#	dxt*=2.
			#else:
			#	flag = 1


		dxt = np.abs(xx-x[i])
		dxt = dxt.max()
		#wx = np.zeros(len(xx),dtype=np.float32)
		#wx = (1.-(np.abs( np.abs( (xx-x[i]) -1 )/dxt))**3)**3
		wx = (1.-(np.abs(xx-x[i])/dxt)**3)**3
		nj = len(wx)

		y_l[i] = 0.0
		if(nj>1):
			for j in range(nj):
				#if(yy[j]!=max(yy)):
					y_l[i] += wx[j]*yy[j]
					w_l[i] += wx[j]
		else:
			y_l[i] = y[i]
			w_l[i] = 1.
		y_l[i] = y_l[i]/w_l[i]
		#print(i,nj,y_l[i],min(wx),max(wx),max(yy))



	return y_l

#USE THIS ONE
#avoids truncation issues at start and end
def loessc(x,y,dx=None):
	#dx = 0.1*(max(x)-min(x))
	if dx is None:
		dx = 1.0e-3*np.abs(max(x)-min(x))
	y_l = np.zeros(len(y),dtype=np.float32)
	w_l = np.zeros(len(y),dtype=np.float32)

	for i in range(len(x)):
		flag = 0
		dxt = dx
		#while(flag==0):
		idx = (np.abs(x-x[i])<dxt).nonzero()
		xx = x[idx]
		yy = y[idx]

			#if(len(xx)<10):
			#	dxt*=2.
			#else:
			#	flag = 1


		#wx = np.zeros(len(xx),dtype=np.float32)
		wx = (1.-(np.abs(xx-x[i])/dxt)**3)**3
		nj = len(wx)

		y_l[i] = 0.0
		if(nj>0):
			for j in range(nj):
				#if(yy[j]!=max(yy)):
					y_l[i] += wx[j]*yy[j]
					w_l[i] += wx[j]
		else:
			y_l[i] = y[i]
			w_l[i] = 1.
		y_l[i] = y_l[i]/w_l[i]
		#print(i,nj,y_l[i],min(wx),max(wx),max(yy))

		if((x[i]-x.min())<dx):
			dxta = np.abs(x[i]-x.min())
			idx = (np.abs(x-x[i])<dxta).nonzero()
			xx = x[idx]
			yy = y[idx]
			wx = (1.-(np.abs(xx-x[i])/dxta)**3)**3
			nj = len(wx)

			y_l[i] = 0.0
			w_l[i] = 0.0
			if(nj>0):
				for j in range(nj):
					#if(yy[j]!=max(yy)):
					y_l[i] += wx[j]*yy[j]
					w_l[i] += wx[j]
			else:
				y_l[i] += y[i]
				w_l[i] += 1.

			y_l[i] = y_l[i]/w_l[i]

		if((x.max()-x[i])<dx):
			dxta = np.abs(x.max()-x[i])
			idx = (np.abs(x-x[i])<dxta).nonzero()
			xx = x[idx]
			yy = y[idx]
			wx = (1.-(np.abs(xx-x[i])/dxta)**3)**3
			nj = len(wx)

			y_l[i] = 0.0
			w_l[i] = 0.0
			if(nj>0):
				for j in range(nj):
					#if(yy[j]!=max(yy)):
					y_l[i] += wx[j]*yy[j]
					w_l[i] += wx[j]
			else:
				y_l[i] += y[i]
				w_l[i] += 1.

			y_l[i] = y_l[i]/w_l[i]
	return y_l










#USE THIS ONE
#avoids truncation issues at start and end
def loessd(x,y,dx=None):
    #dx = 0.1*(max(x)-min(x))
    if dx is None:
        dx = 1.0e-3*np.abs(max(x)-min(x))
    y_l = np.zeros(len(y),dtype=np.float32)
    w_l = np.zeros(len(y),dtype=np.float32)

    for i in range(len(x)):
        flag = 0
        dxt = dx
		#while(flag==0):
		#idx = (np.abs(x-x[i])<dxt).nonzero()
        idm = np.abs(x-x[i]).argmin()
        flag = 0
        idmi = idm
        idma = idm
        while(flag==0):
          if(np.abs(x[idma]-x[i])>dxt):
            flag = 1
          else:
            idma+=1
            if(idma>=len(x)-1):
              idma = len(x)-1
              flag = 1
        flag = 0  
        while(flag==0):
          if(np.abs(x[idmi]-x[i])>dxt):
            flag = 1
          else:
            idmi-=1
            if(idmi<=0):
              idmi = 0
              flag = 1
        idx = np.arange(idmi,idma+1)
          
          
        xx = x[idx]
        yy = y[idx]

			#if(len(xx)<10):
			#	dxt*=2.
			#else:
			#	flag = 1


		#wx = np.zeros(len(xx),dtype=np.float32)
        wx = (1.-(np.abs(xx-x[i])/dxt)**3)**3
        nj = len(wx)

        y_l[i] = 0.0
        if(nj>0):
            for j in range(nj):
                y_l[i] += wx[j]*yy[j]
                w_l[i] += wx[j]
        else:
            y_l[i] = y[i]
            w_l[i] = 1.
        y_l[i] = y_l[i]/w_l[i]
		#print(i,nj,y_l[i],min(wx),max(wx),max(yy))

        if((x[i]-x.min())<dx):
            dxta = np.abs(x[i]-x.min())
            idx = (np.abs(x-x[i])<dxta).nonzero()
            xx = x[idx]
            yy = y[idx]
            wx = (1.-(np.abs(xx-x[i])/dxta)**3)**3
            nj = len(wx)

            y_l[i] = 0.0
            w_l[i] = 0.0
            if(nj>0):
                for j in range(nj):
                    #if(yy[j]!=max(yy)):
                    y_l[i] += wx[j]*yy[j]
                    w_l[i] += wx[j]
            else:
                y_l[i] += y[i]
                w_l[i] += 1.

            y_l[i] = y_l[i]/w_l[i]

        if((x.max()-x[i])<dx):
            dxta = np.abs(x.max()-x[i])
            idx = (np.abs(x-x[i])<dxta).nonzero()
            xx = x[idx]
            yy = y[idx]
            wx = (1.-(np.abs(xx-x[i])/dxta)**3)**3
            nj = len(wx)

            y_l[i] = 0.0
            w_l[i] = 0.0
            if(nj>0):
                for j in range(nj):
                    y_l[i] += wx[j]*yy[j]
                    w_l[i] += wx[j]
            else:
                y_l[i] += y[i]
                w_l[i] += 1.

            y_l[i] = y_l[i]/w_l[i]
    return y_l










