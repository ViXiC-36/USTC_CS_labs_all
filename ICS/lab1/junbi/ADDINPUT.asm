.ORIG X3000
LD R6, ASCII
LD R5, NEGASCII
TRAP X23                    ;"IN" TO R0
ADD R1, R0, X0              ;MOVE THE FIRST INTEGER TO R1
ADD R1, R1, R5              ;CONVERT THE ASCII TO THE NUMBER
TRAP X23                    ;ANOTHER "IN" TO R0
ADD R0, R0, R5              ;CONVERT THE ASCII TO TEH NUMBER
ADD R2, R0, R1              ;ADD THE TWO INTEGERS
ADD R2, R2, R6              ;CONVERT THE SUM BACK TO THE ASCII
LEA R0, MESG                ;LOAD THE ADDRESS OF THE MESSAGE
TRAP X22                    ;"PUTS"
ADD R0, R2, X0              ;SIMPLY MOVE THE SUM TO R0
TRAP X21                    ;DISPLAY THE SUM
HALT
ASCII .FILL X30             ;MASK TO ADD TO CONVER TO ASCII
NEGASCII .FILL XFFD0        ;NEGATED ASCII MASK(-X30)
MESG .STRINGZ "THE SUM IS "
.END