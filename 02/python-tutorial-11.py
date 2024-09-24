# My_Code :  
#import <module_name>
from prnn_package import pricing
from prnn_package.pricing import get_net_price, get_tax
from prnn_package.product import get_tax as g

def main():
    # Using 'import pricing'
    net_price = pricing.get_net_price(price=100, tax_rate=0.01)
    print("Net price using 'import pricing':", net_price)

    # import <module_name> as new_name
    import pricing as selling_price
    net_price = selling_price.get_net_price(price=100, tax_rate=0.01)
    print("Net price using 'import pricing as selling_price':", net_price)

    # from <module_name> import <name>
    net_price = get_net_price(price=100, tax_rate=0.01)
    print("Net price using 'from pricing import get_net_price':", net_price)

    # from <module_name> import <name> as <new_name>
    from pricing import get_net_price as calculate_net_price
    net_price = calculate_net_price(price=100, tax_rate=0.1, discount=0.05)
    print("Net price using 'from pricing import get_net_price as calculate_net_price':", net_price)

    # from <module_name> import * (Demonstrating renaming the tax function)
    tax = g(100)  # This will use the get_tax from product.py
    print("Tax using 'from product import get_tax as gT':", tax)

if __name__ == "__main__":
    main()
