# Decorator functions used to add function attributes 
# such as scrollable / cursor functionality
def scrollable(fn):
    fn.scrollable = True
    return fn

def cursor(fn):
    fn.cursor = True
    return fn