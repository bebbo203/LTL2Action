#True = true
#False = false
#N = Not
#A = And
#O = Or 
#G = Always
#E = Eventually
#X = Next
#U = Until

def progress(formula, assignment):
    '''
        formula: list[list[...]] | List[...]| str
        assignment: list[predicates] 
    '''

    # no operator case
    if type(formula) == str:
        if len(formula) == 1:
            return "True" if formula in assignment else "False"
        # already a bool { "True" | "False" }
        return formula

    
    op = formula[0]

    # AND operator
    if(op == "A"):
        res1 = progress(formula[1], assignment)
        res2 = progress(formula[2], assignment)
        if res1 == "True" and res2 == "True": return "True"
        if res1 == "False" or res2 == "False": return "False"
        if res1 == "True": return res2
        if res2 == "True": return res1
        if res1 == res2:   return res1
        return ["A", res1, res2]
        
    # OR operator
    if(op == "O"):
        res1 = progress(formula[1], assignment)
        res2 = progress(formula[2], assignment)
        if res1 == "True" or res2 == "True": return "True"
        if res1 == "False" and res2 == "False": return "False"
        if res1 == "False": return res2
        if res2 == "False": return res1
        if res1 == res2: return res1
        return ["O", res1, res2]
    
    # NOT operator
    if(op == "N"):
        res = progress(formula[1], assignment)
        if res == "True": return "False"
        if res == "False": return "True"
        return ["N", res]

    # Always operator
    if(op == "G"):
        res = progress(formula[1], assignment)
        if res == "False": return "False"
        if res == "True": return formula
        return ["G", res]

    # Eventually operator
    if(op == "E"):
        res = progress(formula[1], assignment)
        if res == "True": return "True"
        if res == "False": return formula
        return ["E", res]

    # Next operator
    if(op == "X"):
        res = progress(formula[1], assignment)
        return res
    
    # Until operator
    if(op == "U"):
        res1 = progress(formula[1], assignment)
        res2 = progress(formula[2], assignment)
        if res2 == "True": return "True"
        if res1 == "False": return res2
        if res2 == "False":
            if res1 == "True": return formula
            return ["A", res1, formula]
        if res1 == "True":
            if res2 == "False": return formula
            return ["O", res2, formula]
        return ["O", res2, ["A", res1, formula]]



assignment = ["b", "r"]
# formula = ["E", ["A", "b", ["E", "r"]]]
formula = ['A', ['G', ['N', 'b']], ['E', 'r']]
result = progress(formula, assignment)

print(result)


