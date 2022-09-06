from depth_estimator.GLPDepth.code.rw_glp_test import rw_glp_test


"""
return glp model
call do_infer(img_place)

"""
def load_glp():
    model = rw_glp_test()
    return model

def load_adabin():
    pass

def load_binsformer():
    pass


"""
load a particular depth estimator
call do_infer(img_place)

"""
def load_estimator(name='glp'):
    if name == 'glp':
        model = load_glp()

    elif name == 'adabin':
        model = load_adabin()

    else:
        model = load_binsformer()

    return model