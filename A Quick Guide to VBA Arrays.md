# A Quick Guide to VBA Arrays

| Task                               | Static Array                                                 | Dynamic Array                                                |
| :--------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Declare                            | **Dim** arr(0 **To** 5) **As Long**                          | **Dim** arr() **As Long** **Dim** arr **As Variant**         |
| Set Size                           | See Declare above                                            | **ReDim** arr(0 **To** 5)**As Variant**                      |
| Get Size(number of items)          | See [ArraySize](https://excelmacromastery.com/excel-vba-array/#Get_the_Array_Size) function below. | See [ArraySize](https://excelmacromastery.com/excel-vba-array/#Get_the_Array_Size) function below. |
| Increase size (keep existing data) | Dynamic Only                                                 | **ReDim** **Preserve** arr(0 **To** 6)                       |
| Set values                         | arr(1) = 22                                                  | arr(1) = 22                                                  |
| Receive values                     | total = arr(1)                                               | total = arr(1)                                               |
| First position                     | **LBound**(arr)                                              | **LBound**(arr)                                              |
| Last position                      | **Ubound**(arr)                                              | **Ubound**(arr)                                              |
| Read all items(1D)                 | **For** i = **LBound**(arr) **To UBound**(arr) **Next** i Or **For** i = **LBound**(arr,1) **To UBound**(arr,1) **Next** i | **For** i = **LBound**(arr) **To UBound**(arr) **Next** i Or **For** i = **LBound**(arr,1) **To UBound**(arr,1) **Next** i |
| Read all items(2D)                 | **For** i = **LBound**(arr,1) **To UBound**(arr,1)  **For** j = **LBound**(arr,2) **To UBound**(arr,2)  **Next** j **Next** i | **For** i = **LBound**(arr,1) **To UBound**(arr,1)  **For** j = **LBound**(arr,2) **To UBound**(arr,2)  **Next** j **Next** i |
| Read all items                     | **Dim** item **As Variant** **For Each** item **In** arr **Next** item | **Dim** item **As Variant** **For Each** item **In** arr **Next** item |
| Pass to Sub                        | **Sub** MySub(**ByRef** arr() **As String**)                 | **Sub** MySub(**ByRef** arr() **As String**)                 |
| Return from Function               | **Function** GetArray() **As Long**()   **Dim** arr(0 **To** 5) **As Long**   GetArray = arr **End Function** | **Function** GetArray() **As Long**()   **Dim** arr() **As Long**   GetArray = arr **End Function** |
| Receive from Function              | Dynamic only                                                 | **Dim** arr() **As Long** Arr = GetArray()                   |
| Erase array                        | **Erase** arr *Resets all values to default                  | **Erase** arr *Deletes array                                 |
| String to array                    | Dynamic only                                                 | **Dim** arr **As Variant** arr = Split("James:Earl:Jones",":") |
| Array to string                    | **Dim** sName **As String** sName = Join(arr, ":")           | **Dim** sName **As String** sName = Join(arr, ":")           |
| Fill with values                   | Dynamic only                                                 | **Dim** arr **As Variant** arr = Array("John", "Hazel", "Fred") |
| Range to Array                     | Dynamic only                                                 | **Dim** arr **As Variant** arr = Range("A1:D2")              |
| Array to Range                     | Same as dynamic                                              | **Dim** arr **As Variant** Range("A5:D6") = arr              |

 
 

