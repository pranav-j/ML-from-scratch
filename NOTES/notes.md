typedef takes a declaration that would normally create a variable and turns it into a type-name declaration instead.



int x;                    // x is an int variable
typedef int x;            // x is now a type name (alias for int)

int arr[10];              // arr is an array of 10 ints
typedef int arr[10];      // arr is now the type "array of 10 ints"
                          // usage: arr my_array;  ← declares an array

struct Node* next;        // next is a variable
typedef struct Node* NodePtr;  // NodePtr is the type "pointer to struct Node"
                               // usage: NodePtr head;