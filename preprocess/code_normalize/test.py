# import re
code = """
 static int ohci_bus_start(OHCIState *ohci)
 {
     trace_usb_ohci_start(ohci->name);
 
     /* Delay the first SOF event by one frame time as

    if (ohci->eof_timer == NULL) {
        trace_usb_ohci_bus_eof_timer_failed(ohci->name);
        ohci_die(ohci);
        return 0;
    }

    trace_usb_ohci_start(ohci->name);

    /* Delay the first SOF event by one frame time as
 static void ohci_bus_stop(OHCIState *ohci)
 {
     trace_usb_ohci_stop(ohci->name);
    timer_del(ohci->eof_timer);
 }
 
 /* Sets a flag in a port status register but only set it if the port is
}
"""

# annotations = re.findall('(?<!:)\\/\\/.*|\\/\\*(?:\\s|.)*?\\*\\/', code)
# print(annotations)

import re

# 定义正则表达式
pattern = r'//.*|/\*[\s\S]*?\*/'

# 示例代码
# code = """
# // 这是单行注释
# int main() {
#     /* 这是多行注释
#        可以跨越多行 */
#     int x = 5; // 变量声明后的注释
#     return 0;
# }
# """

# 查找所有注释
annotations = re.findall(pattern, code)

# 打印结果
for annotation in annotations:
    print(annotation)
