
# coding: utf-8

# # Create Mortgage Amortization Schedule

# ## Packages

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
# ### Compute sum of the first n terms of a geometric sequence       
#    
# Implement `sum_geo_seq()`. You need to compute $S_n = a \frac{1 - r^n}{1 - r}$.

# In[1]:

# Compute Sn using above formula. Raise warning "Common ratio cannot be 0" if r = 0.
def sum_geo_seq(a,r,n):

    """
    Compute sum of the first n terms of a geometric sequence

    Arguments:
    a -- Scale factor, equal to the sequence's start value
    r -- r ≠ 0 is the common ratio
    n -- Number of terms
    
    Return:
    Sn -- sum_geo_seq(a,r,n)
    """
    

    if r == 0:
        print("Common ratio cannot be 0")
    elif r == 1:
        Sn = n * a
        return Sn
    else:
        Sn = a * (1 - r**n) / (1 - r)
        return Sn    
  


# In[2]:

# Print out result.

a, r, n = (4, 2, 3)
Sn = sum_geo_seq(a,r,n)

print("A geometric sequence has starting value = %.2f, common ratio = %.2f. The sum of its first %d terms = %.2f"
     %(a, r, n, Sn))


a, r, n = (4, 1, 3)
Sn = sum_geo_seq(a,r,n)

print("A geometric sequence has starting value = %.2f, common ratio = %.2f. The sum of its first %d terms = %.2f"
     %(a, r, n, Sn))


a, r, n = (4, 0, 3)
sum_geo_seq(a,r,n)


# **Expected Output**:    
#     
# A geometric sequence has starting value = 4.00, common ratio = 2.00. The sum of its first 3 terms = 28.00   
# A geometric sequence has starting value = 4.00, common ratio = 1.00. The sum of its first 3 terms = 12.00   
# Common ratio cannot be 0   

# In[3]:

# Compute Sn using for loop. Raise warning "Common ratio cannot be 0" if r = 0.
def sum_geo_seq2(a,r,n):

    """
    Compute sum of the first n terms of a geometric sequence

    Arguments:
    a -- Scale factor, equal to the sequence's start value
    r -- r ≠ 0 is the common ratio
    n -- Number of terms
    
    Return:
    Sn -- sum_geo_seq2(a,r,n)
    """    
    
   
    if r == 0:
        print("Common ratio cannot be 0")  
    
    else:
        Sn = 0
        for i in range(n):
            a_i = a * r ** i
            Sn += a_i
        return Sn    


# In[4]:

# Print out result.
a, r, n = (4, 2, 3)
Sn = sum_geo_seq2(a,r,n)
print("A geometric sequence has starting value = %.2f, common ratio = %.2f. The sum of its first %d terms = %.2f"
     %(a, r, n, Sn))

a, r, n = (4, 1, 3)
Sn = sum_geo_seq2(a,r,n)
print("A geometric sequence has starting value = %.2f, common ratio = %.2f. The sum of its first %d terms = %.2f"
     %(a, r, n, Sn))

a, r, n = (4, 0, 3)
sum_geo_seq2(a,r,n)


# **Expected Output**:    
#     
# A geometric sequence has starting value = 4.00, common ratio = 2.00. The sum of its first 3 terms = 28.00   
# A geometric sequence has starting value = 4.00, common ratio = 1.00. The sum of its first 3 terms = 12.00   
# Common ratio cannot be 0   

# In[5]:

# Compute Sn using np.geomspace. Raise warning "Common ratio cannot be 0" if r = 0.
# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.geomspace.html
def sum_geo_seq3(a,r,n):
    
    """
    Compute sum of the first n terms of a geometric sequence

    Arguments:
    a -- Scale factor, equal to the sequence's start value
    r -- r ≠ 0 is the common ratio
    n -- Number of terms
    
    Return:
    Sn -- sum_geo_seq3(a,r,n)
    """    
    

    if r == 0:
        print("Common ratio cannot be 0")
    
    else:
        Sn = np.geomspace(a, a*r**(n-1), num=n).sum()    
        return Sn    


# In[6]:

# Print out result.
a, r, n = (4, 2, 3)
Sn = sum_geo_seq3(a,r,n)
print("A geometric sequence has starting value = %.2f, common ratio = %.2f. The sum of its first %d terms = %.2f"
     %(a, r, n, Sn))

a, r, n = (4, 1, 3)
Sn = sum_geo_seq3(a,r,n)
print("A geometric sequence has starting value = %.2f, common ratio = %.2f. The sum of its first %d terms = %.2f"
     %(a, r, n, Sn))

a, r, n = (4, 0, 3)
sum_geo_seq3(a,r,n)


# **Expected Output**:    
#     
# A geometric sequence has starting value = 4.00, common ratio = 2.00. The sum of its first 3 terms = 28.00   
# A geometric sequence has starting value = 4.00, common ratio = 1.00. The sum of its first 3 terms = 12.00   
# Common ratio cannot be 0   

# ### Compute monthly mortgage payment       
#     
# $ PMT = \frac{BAL}{D} $, where $ D = \sum_{i=1}^n (\frac{1}{1+r})^i $

# In[7]:

def mth_mtg_pmt(Bal, Rate, Period):
    
    """
    Compute monthly mortgage payment

    Arguments:
    Bal    -- Initial balance or Current outstanding balance
    Rate   -- Annual interest rate in percentage
    Period -- Amortization period or Remaining amortization period 
    
    Return:
    Pmt -- mth_mtg_pmt(Bal, Rate, Period)
    """
    
    #Calculate monthly interest rate

    mRate = Rate/12/100

    
    #Calculate monthly mortgage payment using above defined helper function
 
    a = 1 / (1 + mRate)
    r = 1 / (1 + mRate)
    n = Period * 12
    D = sum_geo_seq(a,r,n)
    
    Pmt = Bal / D

    return Pmt


# In[8]:

# Print out result.

Bal, Rate, Period = (760000, 2.19, 25)
Pmt = mth_mtg_pmt(Bal = 760000, Rate = 2.19, Period = 25)

print("When balance = " + str(Bal) + ", annual interest rate = " + str(Rate) + 
      "%, amortization period = " + str(Period) + ",")
print("monthly mortgage payment is: " + str(round(Pmt,2)))


# **Expected Output**:    
#     
# When balance = 760000, annual interest rate = 2.19%, amortization period = 25,    
# monthly mortgage payment is: 3292.06    

# ## Create mortgage amortization schedule

# In[9]:

def amort(Bal, Rate, Period):

    """
    Create mortgage amortization schedule

    Arguments:
    Bal    -- Current outstanding balance
    Rate   -- Annual interest rate in percentage
    Period -- Amortization Period
    
    Returns:
    Res -- Nested list. Mortgage amortization schedule including payment number, outstanding balance, monthly payment,
           interest payment, cumulative interest payment, principal payment, cumulative principal payment
    """
    
    # Parameter initialization

    Res = []
    CumInt = 0
    CumPri = 0

    
    # Calculate monthly mortgage payment based on current balance, 
    # annual interest rate and remaining amortization period
    
    MthPmt = mth_mtg_pmt(Bal, Rate, Period)

    
    # Create mortgage amortization cash flow
   
    for i in range(1, Period * 12 + 1):

    
        # Interest Payment
        # Calculate interest payment as of ith mortgage payment date
       
        IntPmt = Bal * Rate/100/12
 
        # Calculate cumulative interest payment as of ith mortgage payment date
     
        CumInt = CumInt + IntPmt
 

        # Principal Payment
        # Calculate principal payment as of ith mortgage payment date
        
        PriPmt = MthPmt - IntPmt

        # Calculate cumulative principal payment as of ith mortgage payment date

        CumPri = CumPri + PriPmt

        
        # Outstanding Balance
        # Calculate outstanding banlance as of ith mortgage payment date
 
        Bal = Bal - PriPmt
   

        # Create mortgage payment cash flow
        # i, Bal, MthPmt, IntPmt, CumInt, PriPmt, CumPri

        Res.append([i, Bal, MthPmt, IntPmt, CumInt, PriPmt, CumPri])
   
    return Res


# In[10]:

# Print out first two payments given balance = 760000, anuual interest rate is 2.19%, amortization period is 25 years.

CashFlow = amort(Bal = 760000, Rate = 2.19, Period = 25)

print(type(CashFlow))
print('First Payment:', [int(i) for i in CashFlow[0]]) #list comprehension
print('Second Payment:', [int(i) for i in CashFlow[1]])


# **Expected Output**:   
# class 'list'    
# First Payment: [1, 758094, 3292, 1387, 1387, 1905, 1905]      
# Second Payment: [2, 756186, 3292, 1383, 2770, 1908, 3813]

# ## 4 - Convert list of cash flow into pandas DataFrame

# In[11]:

# pandas documentation http://pandas.pydata.org/pandas-docs/stable/index.html
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html
def to_Pandas(Res, FirstPmtDate, Period):
    
    '''
    Create mortgage amortization schedule in Pandas DataFrame format

    Arguments:
    Res -- result from amort(Bal, Rate, Period)
    FirstPmtDate -- String. yyyymmdd.
    Period -- Amortization Period.
    
    Returns:
    Df -- Pandas DataFrame. Complete Mortgage amortization schedule.
    '''
    
    # Convert mortgage amortization schedule from nested list to Pandas DataFrame and 
    # round all columns to 2 decimal places.

    Df = pd.DataFrame(Res).round(2) 

    
    # Add column names
    # 'PmtNo','OutstandingBal','MthPmt','IntPmt','TotIntPaid','PrincipalPmt','TotPrincipalPaid'
    Df.columns = ['PmtNo', 'OutstandingBal', 'MthPmt', 'IntPmt', 'TotIntPaid',
                  'PrincipalPmt', 'TotPrincipalPaid']

    # Add Payment date column

    Df['PmtDate'] = pd.date_range(start=FirstPmtDate, periods=Period*12, freq='MS')
  
    
    # Customize column order
    # 'PmtDate','PmtNo','OutstandingBal','MthPmt','IntPmt','TotIntPaid','PrincipalPmt','TotPrincipalPaid'

    Df = Df[['PmtDate','PmtNo', 'OutstandingBal', 'MthPmt', 'IntPmt', 'TotIntPaid',
               'PrincipalPmt', 'TotPrincipalPaid']]
 
    
    return Df


# In[13]:

# Expected Output:
### START CODE HERE ### (≈ 1 line of code)
Df = to_Pandas(Res = CashFlow, FirstPmtDate = '20170501', Period = 25)
### END CODE HERE ###
Df


# In[13]:

# Export to csv file
# Df.to_csv('mtg_amort_table.csv')


# ## 5 - Plot

# In[14]:

# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html

plt.plot(Df['PmtNo'], Df['TotIntPaid'], label='CumulativeInterestPaid')
plt.plot(Df['PmtNo'], Df['TotPrincipalPaid'], label='CumulativePrincipalPaid')
plt.plot(Df['PmtNo'], Df['OutstandingBal'], label='OutstandingBalance')
plt.legend()

plt.show()


# In[15]:

# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.stackplot.html
# https://matplotlib.org/gallery/lines_bars_and_markers/stackplot_demo.html

plt.stackplot(Df['PmtNo'], Df['PrincipalPmt'], Df['IntPmt'], labels=['PrincipalPayment','InterestPayment'])
plt.legend(loc='upper left')

plt.show()


# ## Part 4 - Advanced: Prepayment

# In[16]:

def adv_amort(Bal, Rate, Period, PrePmt = {}):
    
    """
    Create mortgage amortization schedule

    Arguments:
    Bal    -- Current outstanding balance
    Rate   -- Annual interest rate in percentage
    Period -- Amortization Period
    PrePmt -- dict, Optional. 
              Prepayment between ith and (i+1)th payment
    
    Returns:
    Res -- Nested list. Mortgage amortization schedule including payment number, outstanding balance, monthly payment,
           interest payment, cumulative interest payment, principal payment, cumulative principal payment,
           prepayment amount, cumulative prepayment amount
    """
    
    # Parameter initialization

    Res = []
    CumInt = 0
    CumPri = 0
    CumPre = 0

    
    # Calculate monthly mortgage payment based on current balance, 
    # annual interest rate and remaining amortization period

    MthPmt = mth_mtg_pmt(Bal, Rate, Period)

    
    # Create mortgage amortization cash flow

    for i in range(1, Period * 12 + 1):

    
        # Interest Payment
        # Calculate interest payment as of ith mortgage payment date
  
        IntPmt = Bal * Rate/100/12
  
        
        # Calculate cumulative interest payment as of ith mortgage payment date
 
        CumInt = CumInt + IntPmt

        # Prepayment
        # Set prepayment value if there is a prepayment on ith mortgage payment date, set to 0 otherwise.
 
        if i in PrePmt:
            PrePmtVal = PrePmt[i]
        else:
            PrePmtVal = 0


        # Raise warning "Prepayment amount cannot exceed current balance! Please re-enter."
        # if prepayment amount is more than current balance and stop creating the cash flow.
        
        if PrePmtVal > Bal:
            print('Prepayment amount cannot exceed current balance! Please re-enter')
            break

        # Calculate cumulative prepayment as of ith mortgage payment date
       
        CumPre = CumPre + PrePmtVal
   
        
        # Principal Payment
        # Calculate principal payment as of ith mortgage payment date
        
        PriPmt = MthPmt - IntPmt + PrePmtVal
    

        # Outstanding Balance
        # Calculate outstanding banlance as of ith mortgage payment date
  
        Bal = Bal - PriPmt


        # Adjust payment amount on ith mortgage payment date if outstanding balance is negative after ith payment
   
        if Bal < 0:
            Bal = Bal + PriPmt
            MthPmt = IntPmt + Bal
            PriPmt = Bal
            CumPri = CumPri + PriPmt
            Bal = 0
            Res.append([i, Bal, MthPmt, IntPmt, CumInt, PriPmt, CumPri, PrePmtVal, CumPre])
            break
     
        
        # Calculate cumulative principal payment as of ith mortgage payment date
     
        CumPri = CumPri + PriPmt

        
        # Create mortgage payment cash flow
        # i, Bal, MthPmt, IntPmt, CumInt, PriPmt, CumPri, PrePmtVal, CumPre

        Res.append([i, Bal, MthPmt, IntPmt, CumInt, PriPmt, CumPri, PrePmtVal, CumPre])       

        
    return Res


# In[17]:

# Print out first four payments given balance = 760000, anuual interest rate is 2.19%, 
# amortization period is 25 years and 
# prepayments were made on the third and fifth payment date with 300000 and 100000 dollars.

adv_CashFlow = adv_amort(Bal = 760000, Rate = 2.19, Period = 25
                                           , PrePmt = {3:300000,5:100000})

print('First Payment:', [int(i) for i in adv_CashFlow[0]])
print('Second Payment:', [int(i) for i in adv_CashFlow[1]])
print('Third Payment:', [int(i) for i in adv_CashFlow[2]])
print('Forth Payment:', [int(i) for i in adv_CashFlow[3]])


# **Expected Output**:   
# First Payment: [1, 758094, 3292, 1387, 1387, 1905, 1905, 0, 0]   
# Second Payment: [2, 756186, 3292, 1383, 2770, 1908, 3813, 0, 0]   
# Third Payment: [3, 454274, 3292, 1380, 4150, 301912, 305725, 300000, 300000]   
# Forth Payment: [4, 451811, 3292, 829, 4979, 2463, 308188, 0, 300000]    

# In[19]:

def adv_to_Pandas(Res, FirstPmtDate, Period):
    
    '''
    Create mortgage amortization schedule in Pandas DataFrame format

    Arguments:
    Res -- result from adv_amort(Bal, Rate, Period)
    FirstPmtDate -- String. yyyymmdd.
    Period -- Amortization Period.
    
    Returns:
    Df -- Pandas DataFrame. Complete Mortgage amortization schedule.
    '''
    
    # Convert mortgage amortization schedule from nested list to Pandas DataFrame and 
    # round all columns to 2 decimal places.

    Df = pd.DataFrame(Res).round(2) 

     
    # Add column names
    # 'PmtNo','OutstandingBal','MthPmt','IntPmt','TotIntPaid','PrincipalPmt','TotPrincipalPaid','PrePmt','TotPrePmtPaid'

    Df.columns = (['PmtNo', 'OutstandingBal', 'MthPmt', 'IntPmt', 'TotIntPaid',
                    'PrincipalPmt', 'TotPrincipalPaid', 'PrePmt', 'TotPrePmtPaid'])

    
    # Add Payment date column    

    Df['PmtDate'] = pd.date_range(start=FirstPmtDate, periods=Df.shape[0], freq='MS')

    # Customize column order
    # 'PmtDate','PmtNo','OutstandingBal','MthPmt','IntPmt','TotIntPaid','PrincipalPmt','TotPrincipalPaid','PrePmt','TotPrePmtPaid'

    Df = Df[['PmtDate','PmtNo', 'OutstandingBal', 'MthPmt', 'IntPmt', 'TotIntPaid',
               'PrincipalPmt', 'TotPrincipalPaid', 'PrePmt', 'TotPrePmtPaid']]

    
    return Df


# In[20]:

# Expected Output:

adv_to_Pandas(Res = adv_CashFlow, FirstPmtDate = '20170501', Period = 25)

