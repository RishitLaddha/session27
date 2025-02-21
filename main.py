import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import custom_cnn
import custom_transformer

def main():
    print("========== Running Custom CNN Training ==========")
    custom_cnn.main()  # Now runs for 5 epochs and prints nicely formatted progress.
    
    print("\n========== Running Custom Transformer Training ==========")
    custom_transformer.main()  # Now runs for 5 epochs as well.

if __name__ == "__main__":
    main()
