# input of the data
i1 = ["B", True, True, True]
i2 = ["B", True, True, True]
i3 = ["B", False, False, True]
i4 = ["T", False, False, True]
i5 = ["T", True, True, False]
i6 = ["T", True, False, False]

input = [i1, i2, i3, i4, i5, i6]

BC = 0 # Bronchitis count
TC = 0 # Tuberculosis count
XCB = 0 # X-ray shadow count for Bronchitis
XCT = 0 # X-ray shadow count for Tuberculosis
DCB = 0 # Dyspnea count for Bronchitis
DCT = 0 # Dyspnea count for Tuberculosis
LCB = 0 # Lung inflammation count for Bronchitis
LCT = 0 # Lung inflammation count for Tuberculosis

for i in range(0, len(input)):
    curInput = input[i]
    if (curInput[0] == "B"):
        BC = BC + 1

        if (curInput[1]):
            XCB = XCB + 1

        if (curInput[2]):
            DCB = DCB + 1

        if (curInput[3]):
            LCB = LCB + 1
    else: # curInput[0] == "T"
        TC = TC + 1

        if (curInput[1]):
            XCT = XCT + 1

        if (curInput[2]):
            DCT = DCT + 1

        if (curInput[3]):
            LCT = LCT + 1

PrXBY = XCB * 1.0 / BC # probability of X-ray shadow Yes given by Bronchitis
PrXBN = 1 - PrXBY # probability of X-ray shadow No given by Bronchitis
PrXTY = XCT * 1.0 / TC # probability of X-ray shadow Yes given by Tuberculosis
PrXTN = 1 - PrXTY # probability of X-ray shadow No given by Tuberculosis

PrDBY = DCB * 1.0 / BC # probability of Dyspnea Yes given by Bronchitis
PrDBN = 1 - PrDBY # probability of Dyspnea No given by Bronchitis
PrDTY = DCT * 1.0 / TC # probability of Dyspnea Yes given by Tuberculosis
PrDTN = 1 - PrDTY # probability of Dyspnea No given by Tuberculosis

PrLBY = LCB * 1.0 / BC # probability of Lung inflammation Yes given by Bronchitis
PrLBN = 1- PrLBY # probability of Lung inflammation No given by Bronchitis
PrLTY = LCT * 1.0 / TC # probability of Lung inflammation Yes given by Tuberculosis
PrLTN = 1 - PrLTY # probability of Lung inflammation No given by Tuberculosis

PrB = BC * 1.0 / (BC + TC) # probability of Bronchitis
PrT = TC * 1.0 / (BC + TC) # probability of Tuberculosis

print("P(X-ray Shadow|Bronchitis) = {0}".format(PrXBY))
print("P(Dyspnea|Bronchitis) = {0}".format(PrDBY))
print("P(Lung inflammation|Bronchitis) = {0}".format(PrLBY))
print("P(X-ray Shadow|Tuberculosis) = {0}".format(PrXTY))
print("P(Dyspnea|Tuberculosis) = {0}".format(PrDTY))
print("P(Lung inflammation|Tuberculosis) = {0}".format(PrLTY))
print("P(Bronchitis) = {0}".format(PrB))
print("P(Tuberculosis) = {0}".format(PrT))

# according to the NB assumption that all features are independent
PrIfB = PrB * PrXBY * PrDBN * PrLBY
PrIfT = PrT * PrXTY * PrDTN * PrLTY

print("The patient is likely to be Bronchitis with probability {0}".format(PrIfB))
print("The patient is likely to be Tuberculosis with probability {0}".format(PrIfT))

if (PrIfB > PrIfT):
    print("This patient is mostly likely to have Bronchitis with probability {0}".format(PrIfB))
else :
    print("This patient is mostly likely to have Tuberculosis with probability {0}".format(PrIfT))

# This learned model is likely to be overfit since all the features are not independent, there are actually
# related to each other. Since for some disease, patient is more likely to have some symtoms together

