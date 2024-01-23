def instruction():
    print("How this program works?")
    print("1. Enter the input(example: new york)")
    print("2. It will show you the Japanese transliteration and FST diagram window")
    print("3. Close the FST diagram window and it will ask you either to continue / not")
    print("4. If you type 1, the process will loop and again you have to enter the input")
    print("5. The output will be appended in Jap-trans.dat file")
    print("6. The program stop when you wish not to continue by passing 0")
    print("*******************************************************************")

# example that uses the nltk_contrib FST class
from nltk.nltk_contrib.fst.fst import *

class myFST(FST):
    def recognize(self, iput, oput):

        # insert your codes here
        # self.inp = iput.split()
        # self.outp = oput.split()
        self.inp = list(iput)
        self.outp = list(oput)
        # f.transduce("abc")

        if list(oput) == f.transduce(list(iput)):
            # print(" ".join(f.transduce(iput.split())))
            return True
        else:
            return False

f = myFST('example')
# first add the states in the FST
for i in range(1, 6):
    f.add_state(str(i))  # add states '1' .. '5'

# add one initial state
f.initial_state = '1'  # -> 1

# add all transitions
f.add_arc('1', '2', ['com'], ['kon'])  # 1 -> 2 [com:kon]
f.add_arc('1', '2', ['new'], ['nyuu'])  # 1 -> 2 [new:nyuu]
f.add_arc('1', '2', ['bill'], ['biru'])  # 1 -> 2 [bill:biru]
f.add_arc('1', '2', ['golf'], ['goruhu'])  # 1 -> 2 [golf:goruhu]
f.add_arc('1', '2', ['te'], ['te'])  # 1 -> 2 [te:te]
f.add_arc('1', '2', ['ra'], ['ra'])  # 1 -> 2 [ra:ra]
f.add_arc('1', '2', ['sto'], ['sutoo'])  # 1 -> 2 [sto:sutoo]
f.add_arc('1', '2', ['twin'], ['tsuin'])  # 1 -> 2 [twin:tsuin]
f.add_arc('1', '2', ['video'], ['bideo'])  # 1 -> 2 [video:bideo]
f.add_arc('1', '2', ['res'], ['resu'])  # 1 -> 2 [res:resu]
f.add_arc('1', '2', ['ele'], ['ere'])  # 1 -> 2 [ele:ere]
f.add_arc('1', '2', ['ice'], ['aisu'])  # 1 -> 2 [ice:aisu]

f.add_arc('2', '2', ['#'], ['#'])  # 2 -> 2 [#:#]

f.add_arc('2', '3', ['pu'], ['pyu'])  # 2 -> 3 [pu:pyu]
f.add_arc('2', '3', ['le'], ['re'])  # 2 -> 3 [le:re]
f.add_arc('2', '3', ['tau'], ['to'])  # 2 -> 3 [tau:to]
f.add_arc('2', '3', ['va'], ['bee'])  # 2 -> 3 [va:bee]

f.add_arc('3', '3', ['#'], ['#'])  # 3 -> 3 [#:#]

f.add_arc('3', '4', ['ter'], ['taa'])  # 3 -> 4 [ter:taa]
f.add_arc('3', '4', ['rant'], ['ran'])  # 3 -> 4 [rant:ran]
f.add_arc('3', '4', ['tor'], ['ta'])  # 3 -> 4 [tor:ta]

f.add_arc('2', '4', ['york'], ['yooku'])  # 2 -> 4 [york:yooku]
f.add_arc('2', '4', ['gates'], ['geitsu'])  # 2 -> 4 [gates:geitsu]
f.add_arc('2', '4', ['ball'], ['booru'])  # 2 -> 4 [ball:booru]
f.add_arc('2', '4', ['dio'], ['jio'])  # 2 -> 4 [dio:jio]
f.add_arc('2', '4', ['ry'], ['rii'])  # 2 -> 4 [ry:rii]
f.add_arc('2', '4', ['tower'], ['tawa'])  # 2 -> 4 [tower:tawa]
f.add_arc('2', '4', ['game'], ['gemu'])  # 2 -> 4 [game:gemu]
f.add_arc('2', '4', ['cream'], ['kuriimu'])  # 2 -> 4 [cream:kuriimu]

f.add_arc('3', '5', ['vi'], ['bi'])  # 3 -> 5 [vi:bi]

f.add_arc('5', '5', ['#'], ['#'])  # 5 -> 5 [#:#]

f.add_arc('5', '4', ['sion'], ['bangu'])  # 5 -> 4 [sion:bangu]

# add final/accepting state(s)
f.set_final('4')  # 4 ->

def check2(inp, outp):
    if f.recognize(inp, outp):
        result = str(inp) + "-->" + str(outp) +"\n"
        with open('Jap-trans.dat', 'a') as file:
            file.write(str(result))
            print(result)
            # print("accept")
            disp = FSTDemo(f)
            # disp.transduce("bill#gates".split('#'))
            disp.mainloop()
    else:
        print("reject")


def check(u):
    if user[0:2]=='co':
        inp = "com#pu#ter".split('#')
        outp = "kon#pyu#taa".split('#')
        check2(inp,outp)
    elif user[0:2]=='ne':
        inp = "new#york".split('#')
        outp = "nyuu#yooku".split('#')
        check2(inp,outp)
    elif user[0:2]=='bi':
        inp = "bill#gates".split('#')
        outp = "biru#geitsu".split('#')
        check2(inp,outp)
    elif user[0:2]=='go':
        inp = "golf#ball".split('#')
        outp = "goruhu#booru".split('#')
        check2(inp,outp)
    elif user[0:2]=='te':
        inp = "te#le#vi#sion".split('#')
        outp = "te#re#bi#bangu".split('#')
        check2(inp,outp)
    elif user[0:2]=='ra':
        inp = "ra#dio".split('#')
        outp = "ra#jio".split('#')
        check2(inp,outp)
    elif user[0:2]=='st':
        inp = "sto#ry".split('#')
        outp = "sutoo#rii".split('#')
        check2(inp,outp)
    elif user[0:2]=='tw':
        inp = "twin#tower".split('#')
        outp = "tsuin#tawa".split('#')
        check2(inp,outp)
    elif user[0:2]=='vi':
        inp = "video#game".split('#')
        outp = "bideo#gemu".split('#')
        check2(inp,outp)
    elif user[0:2]=='re':
        inp = "res#tau#rant".split('#')
        outp = "resu#to#ran".split('#')
        check2(inp,outp)
    elif user[0:2]=='el':
        inp = "ele#va#tor".split('#')
        outp = "ere#bee#ta".split('#')
        check2(inp,outp)
    elif user[0:2]=='ic':
        inp = "ice#cream".split('#')
        outp = "aisu#kuriimu".split('#')
        check2(inp,outp)
    else:
        print("reject")

instruction()
user=input("\nEnter input: ")
check(user[0:2])
cont = input("Do you wish to continue? (1 for yes, 0 for no): ")
while (cont=='1'):
    user = input("Enter input: ")
    check(user[0])
    cont = input("Do you wish to continue? (1 for yes, 0 for no): ")