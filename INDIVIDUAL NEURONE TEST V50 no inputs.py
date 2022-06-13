import pdfplumber
import pickle
import os
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random
import numpy as np

class mind:  
    class neurone:
        def __init__(self):
            
            self.dendrites=[]
            self.weights=[]
            self.responses=[]
            

            self.output=0
            self.error=0

            self.derivative=1


        def connect(self,neurone,value=1):
            import random
            self.dendrites.append(neurone)
            self.weights.append(random.random()*value)
            self.responses.append(0)

        def forward(self):
            new_calc=0
            for neurone in range(len(self.dendrites)):
                new_calc+=self.dendrites[neurone].output*self.weights[neurone]
                self.responses[neurone]=self.dendrites[neurone].output
            try:
                self.output=1/(1 + (2.718281828459045235360**-new_calc))
            except:
                self.output=1
            self.derivative=(self.output * (1 - self.output))
        def set_error(self,error):
            self.error=error*self.derivative
        def backwards(self):
            for i in range(len(self.weights)):
                change=(self.error*self.responses[i])
                if self.derivative !=0:
                    self.weights[i]+=change/self.derivative
                    self.dendrites[i].error+=(self.dendrites[i].derivative*change)*self.weights[i]
            self.error=0
    def __init__(self):
        self.outputs=[]
        self.neurones=[]
        self.inputs=[]
        self.brain_mass=0
        self.input_marker=0
        self.neural_marker=0
        self.relationship_marker=0
    def add_input(self):
        new=self.neurone()
        self.inputs.append(new)
    def add_output(self):
        new=self.neurone()
        self.outputs.append(new)
        self.neurones.append(new)
    def add_neurone(self):
        self.neurones.append(self.neurone())
    def add_relationship(self,connect_to,connect_from,value=1):
        self.neurones[connect_to].connect(self.neurones[connect_from],value=value)
        self.brain_mass+=1
    def add_input_relationship(self,connect_to,connect_from):
        self.neurones[connect_to].connect(self.inputs[connect_from])
        self.brain_mass+=1
    def forward(self,inputs):
        for i in range(len(inputs)):
            self.inputs[i].output=inputs[i]
        for neurone in self.neurones:
            neurone.forward()
        outputs=[]
        for neurone in self.outputs:
            outputs.append(neurone.output)
        return outputs
    def backwards(self,errors,learning_rate=0.1):
        for neurone in range(len(self.outputs)):
            self.outputs[neurone].set_error(errors[neurone]*learning_rate)
        for neurone in reversed(self.neurones):
            neurone.backwards()
    def backwards_print(self,errors,learning_rate=0.1):
        for neurone in self.inputs:
            neurone.error=0
        for neurone in range(len(self.outputs)):
            self.outputs[neurone].set_error(errors[neurone]*learning_rate)
        for neurone in reversed(self.neurones):
            print("error")
            print(neurone.error)
            print("error divided")
            print(neurone.error/len(neurone.dendrites))
            print("error divided **2")
            print(neurone.error/(len(neurone.dendrites)**2))
            neurone.backwards()
        for neurone in self.inputs:
            print(neurone.error)
    def brain_damage(self):
        mini=100000000
        neurone_location=0
        weight_location=0
        for neurone in range(len(self.neurones)):
            for weight in range(len(self.neurones[neurone].weights)):
                if abs(self.neurones[neurone].weights[weight]) < mini:
                    mini=abs(self.neurones[neurone].weights[weight])
                    neurone_location=neurone
                    weight_location=weight_location
        self.neurones[neurone_location].weights.pop(weight_location)
        self.neurones[neurone_location].dendrites.pop(weight_location)
        self.neurones[neurone_location].responses.pop(weight_location)
        self.brain_mass-=1
    def neuro_genesis(self):

        if self.input_marker==len(self.inputs) and self.neural_marker==len(self.neurones):
            print("new neurone")
            self.neurones.insert(0,self.neurone())
            self.input_marker=0
            self.neural_marker=0
            self.relationship_marker=1
        elif self.input_marker==self.neural_marker and self.input_marker < len(self.inputs):
            print("new input wiring")
            self.add_input_relationship(0,self.input_marker)
            self.input_marker+=1
        elif self.relationship_marker==self.neural_marker and self.relationship_marker < len(self.neurones):
            print("relationship added")
            self.add_relationship(self.neural_marker,0)
            self.relationship_marker+=1
        elif self.neural_marker < len(self.neurones):
            print("back wiring added")
            self.add_relationship(0,self.neural_marker)
            self.neural_marker+=1
        else:
            print("new input relationship added")
            self.add_input_relationship(0,self.input_marker)
            self.input_marker+=1
    def len(self,value=0,neurones=[]):
        if len(neurones)==0:
            neurones=self.neurones
        reply=[]
        for neurone in self.neurones:
            if len(neurone.dendrites)==value:
                reply.append(neurone)
        return reply
    def len_greater(self,value=0,neurones=[]):
        if len(neurones)==0:
            neurones=self.neurones
        reply=[]
        for neurone in self.neurones:
            if len(neurone.dendrites)>=value:
                reply.append(neurone)
        return reply
    def len_lower(self,value=0,neurones=[]):
        if len(neurones)==0:
            neurones=self.neurones
        reply=[]
        for neurone in self.neurones:
            if len(neurone.dendrites)<=value:
                reply.append(neurone)
        return reply
    def not_in_graph(self,neurones=[],to_exclude=[]):
        if len(neurones)==0:
            neurones=self.neurones[0]
        reply=[]  
        for neurone in neurones:
            test=self.graphs([neurone])
            check=True
            for i in test:
                for ex in to_exclude:
                    if i is ex:
                        check=False
            if check==True:
                reply.append(neurone)
        return reply
    def graphs(self,neurones=[],visted=[]):
        if len(neurones)==0:
            neurones=self.neurones[0]
        reply=[]
        for neurone in neurones:
            test=self.exclude(neurone.dendrites,visted)
            for nex in test:
                visted.append(nex)
                reply.append(nex)
            test=self.graphs(test,visited)
            for nex in test:
                visted.append(nex)
                reply.append(nex)
        self.clean(reply)
        return reply
    def add_bias(self,value):
        for neurone in self.inputs:
            if neurone.output==value:
                return neurone
        self.inputs(self.neurone())
        self.inputs[-1].output=value
        return self.inputs[-1]
    def exclude(self,neurones=[],compare=[]):
        if len(neurones)==0:
            neurones=self.neurones
        if len(compare)==0:
            compare=self.inputs
        reply=[]
        for neurone in neurones:
            test=True
            for check in compare:
                if neurone is check:
                    test=False
            if test==True:
                reply.append(neurone)
        return reply
    def not_in(self,neurones=[],compare=[]):
        if len(neurones)==0:
            neurones=self.neurones
        if len(compare)==0:
            compare=self.inputs
        reply=[]
        for neurone in neurones:
            test=True
            for check in compare:
                if neurone is check:
                    test=False
            if test==True:
                reply.append(neurone)
        return reply
    def In(self,neurones=[],compare=[]):
        if len(neurones)==0:
            neurones=self.neurones
        if len(compare)==0:
            compare=self.inputs
        reply=[]
        for neurone in neurones:
            for check in compare:
                if neurone is check:
                    reply.append(neurone)
        return reply
    def connected_to(self,neurones=[],compare=[],visited=[]):
        if len(neurones)==0:
            neurones=self.neurones
        if len(compare)==0:
            compare=self.inputs
        reply=[]
        for neurone in neurones:
            check=self.In(neurone.dendrites,compare)
        for test in check:
            reply.append(test)
        return self.clean(reply)
    def clean(self,neurones):
        tests=[True]*len(neurones)
        for i in range(len(neurones)):
            if tests[i]==True:
                for b in range(len(neurones)):
                    if i!=b:
                        if i < b and neurones[i] is neurones[b]:
                            neurones[b]=False
        reply=[]
        for i in range(len(neurones)):
            if tests[i]==True:
                reply.append(neurones[i])
        return reply
    def honey_comb(self,neurones=[],pattern=2,value=1):
        if len(neurones)==0:
            neurones=self.len()
        to_add=[]
        for neurone in neurones:
            to_add.insert(0,self.neurone())
        for neurone in range(len(neurones)):
            for i in range(pattern):
                neurones[neurone].connect(to_add[neurone-i],value)
        for neurone in to_add:
            self.neurones.insert(0,neurone)
        return to_add
    def trepanning(self,skull_size=3906):
        print(self.brain_mass)
        if self.brain_mass > skull_size:
            print("brain damage")
            self.brain_damage()
        else:
            print("mind grew")
            self.neuro_genesis()
    def recurrent(self,neurones,value=1,ranged=0):
        for neurone in range(len(neurones)):
            for connect in range(value):
                neurones[neurone].connect(neurones[neurone-connect],value)
    def inhibitor(self,neurones,value=1):
        for neurone in range(len(neurones)):
            for connect in range(len(neurones)):
                if neurone > connect and ((neurone-connect) < ranged or ranged==0):
                    neurones[neurone].connect(neurones[connect],value)
    def convergence(self,neurones,steps,value=1):
        to_add=[]
        count=0
        while count < len(neurones):
            to_add.insert(0,self.neurone())
            for i in range(steps):
                if count < len(neurones):
                    neurones[count].connect(to_add[0],value)
                count+=1
        for neurone in to_add:
            self.neurones.insert(0,neurone)
        return to_add
    def divergence(self,neurones=[],steps=2,value=0):
        to_add=[]
        if len(neurones)==0:
            neurones=self.len()
        for neurone in neurones:
            for i in range(steps):
                to_add.append(self.neurone())
                if value==0 and (i+1)%2==0:
                    neurone.connect(to_add[-1],value)
                elif value==0 and (i+1)%2!=0:
                    neurone.connect(to_add[-1],-1)
                else:
                    neurone.connect(to_add[-1],value)
        for neurone in to_add:
            self.neurones.insert(0,neurone)
        return to_add
    def fractal(self,neurones=[],iterations=2,divergence=2,value=1):
        if len(neurones)==0:
            neurones=self.neurones
        for i in range(iterations):
            to_add=[]
            for neurone in neurones:
                for i in range(divergence):
                    to_add.insert(0,self.neurone())
                for i in range(divergence):
                    if (i+1)%2==0:
                        neurone.connect(to_add[i],value=-1)
                    else:
                        neurone.connect(to_add[i],value=1)
            for neurone in to_add:
                self.neurones.insert(0,neurone)
            return to_add
    def recurrent_fractal(self,iterations,divergence):
        for i in range(iterations):
            to_add=[]
            for neurone in self.neurones:
                if len(neurone.dendrites)==0:
                    for i in range(divergence):
                        to_add.insert(0,self.neurone())
                        if (i+1)%2==0:
                            neurone.connect(to_add[0],value=-1)
                        else:
                            neurone.connect(to_add[0],value=1)
                        to_add[0].connect(neurone)

                            
            for neurone in reversed(to_add):
                self.neurones.insert(0,neurone)
        for i in range(divergence):
            if i==divergence:
                self.fill_in(self.neurones[-2],[self.neurones[-1-divergence]],pattern=0,a=0,b=2,offste=0,size_targeted=2)
            else:
                self.fill_in(self.neurones[-1-i],[self.neurones[-2-i]],pattern=0,a=0,b=2,offste=0,size_targeted=2)
        for ini in self.inputs:
            self.neurones[-1].connect(ini)
    def block(self,neurones,width,extra_depth,pattern=0,a=0,b=2,offste=0):
        for a in range(width):
            block.append(self.neurone())
        for neurone in neurones:
            block=[]
            for a in range(width):
                block.append(self.neurone())
                neurone.connect(block[a])
            for i in range(extra_depth):
                row=[]
                for a in range(width):
                    row.append(self.neurone())
                for new in block:
                    if len(new.dendrites)==0:
                        sub=0
                        for add in row:
                            if pattern==0:
                                new.connect(add)
                            if pattern==1:
                                a2=(a+sub)%len(neurones)
                                b2=(b+sub)%len(neurones)
                                if (c > a2 and c < b2):
                                    new.connect(add)
                            sub+=1
                for add in row:
                    block.append(add)
        for neurone in block:
            self.neurones.insert(0,neurone)
        return row
    def finalise_inputs(self,neurones,to_add=[],value=1):
        for neurone in neurones:
            for ini in self.inputs:
                neurone.connect(ini,value)
            for add in to_add:
                neurone.connect(add,value)
    def propogate_neurone(self,start,local,pattern=0,a=0,b=2,offste=0):
        propogation=[self.neurones[local]]
        for ini in self.inputs:
            propogation.append(ini)
        self.fill_in(self.neurones[start],propogation,pattern=0,a=0,b=2,offste=0)
    def propogate_lines(self,start,local,pattern=0,a=0,b=2,offste=0):
        propogation=[]
        for i in range(local):
            propogation.append(self.neurone())
        self.fill_in(self.neurones[start],propogation,pattern=0,a=0,b=2,offste=0)
    def propogate_recursion(self,start,local,pattern=0,a=0,b=2,offste=0):
        propogation=[]
        for i in local:
            propogation.append(self.neurones[i])
        self.fill_in(self.neurones[start],propogation,pattern=0,a=0,b=2,offste=0)
    def add_neurones(self,neurones,to_add=[],value=1,split=False,random_forest=False):
        if len(neurones)==0:
            neurones=self.len()
        if len(to_add)==0:
            to_add=self.inputs
        if random_forest==True:
            import random
            new=[]
            for neurone in neurones:
                if random.random() > 0.5:
                    new.append(neurone)
            neurones=new
        if split==True:
            check_add=False
            check_neurones=False
            rail_add=0
            rail_neurones=0
            while check_add==False or check_neurones==False:
                if value==0:
                    if (rail_neurones+1)%2==0:
                        neurones[rail_neurones%len(neurones)].connect(to_add[rail_add%len(to_add)],1)
                    else:
                        neurones[rail_neurones%len(neurones)].connect(to_add[rail_add%len(to_add)],-1)
                else:
                    neurones[rail_neurones%len(neurones)].connect(to_add[rail_add%len(to_add)],value)
                rail_add+=1
                rail_neurones+=1
                if rail_neurones%len(neurones)==0:
                    check_neurones=True
                if rail_add%len(to_add)==0:
                    check_add=True

        else:
            for neurone in neurones:
                for check in to_add:
                    neurone.connect(check,value)
    def fill_in(self,neurone,propogation,pattern=0,a=0,b=2,offste=0,size_targeted=0,visited=[]):
        c=0
        for neu in neurone.dendrites:
            for check in visited:
                if check is neu:
                    return
            visited.append(neu)
            if len(neu.dendrites)<=size_targeted:
                sub=0
                for prop in propogation:
                    if pattern==0:
                        neu.connect(prop)
                    elif pattern==1:
                        a2=(a+sub)%len(neurones)
                        b2=(b+sub)%len(neurones)
                        if (c > a2 and c < b2):
                            neu.connect(prop)
                    elif pattern==2:
                        a2=(a+sub)%len(neurones)
                        if c > a2:
                            neu.connect(prop)
                    elif pattern==3:
                        b2=(b+sub)%len(neurones)
                        if c < b2:
                            neu.connect(prop)
                    elif pattern==4:
                        b2=(sub)%b
                        if b2==0:
                            neu.connect(prop)

                    sub+=1
                
            else:
                self.fill_in(neu,propogation,pattern=0,a=0,b=2,offste=0)
            c+=1
                    
                    
                
            
        
                        

        
def create_network(depth,fractal):
    brain=mind()
    brain.add_output()
    for i in range(start_pad):
        brain.add_neurone()
    for i in range(inputs):
        brain.add_input()
        brain.add_neurone()
    for i in range(end_pad):
        brain.add_neurone()
    for i in range(outputs):
        brain.add_output()
    a=len(brain.neurones)
    d=len(brain.inputs)
    for b in range(a):
        for c in range(d):
            brain.add_input_relationship(b,c)
        for c in range(a):
            if b !=c:
                brain.add_relationship(b,c)
    return brain
def train_once(number,mind,file='kjv_bible_with_apocrypha.pdf',path = r"C:\Data\Novels"):
    #if os.path.exists(r'C:\Data\jabberrer.pkl'):
    #    GENE=open(r'C:\Data\jabberrer.pkl','rb')
    #    unpickler = pickle.Unpickler(GENE)
    #    left_mind = unpickler.load()
    #    GENE.close()
    #else:
    sized=[]
    patterns=[]


    # specify your path of directory

    
     
    # call listdir() method
    # path is a directory of which you want to list
    directories = os.listdir( path )
    inputs=[0]*len(numbers)
    #check=mind.forward(inputs)
    check=mind.forward([])
    evaluate=(1/len(numbers))
    scored=[]
    pages=[]
    check1=0.5
    check2=0.5
    check=0

    errors=[]
    page_count=[]
    

    with pdfplumber.open(path+"\\"+file) as pdf:
        num_of_pages = len(pdf.pages)
        a_eval=[]
        b_eval=[]
        c_eval=0
        count=0
        d_eval=10000000000000
        page_point=0
        pdf=pdf.pages
        page_ranges = len(pdf)
        word_champ=''
        word_best=1
        word_list={}
        word_step={}
        mean_1=0.5
        mean_2=0.5
        count=0
        score=0
        word_eval=0
        word_count=0
        word_considered=''
        for page in pdf:
            #mind.trepanning()
            try:
                errors.append(score/count)
            except:
                errors.append(0)
            page_count.append(page_point)
            score=0
            count=0
            print(c_eval)
            page_point+=1
            printin=page.extract_text()
            mean=0
            if printin !=None:
                for p in printin:
                    if p.lower() in switch:
                        si=switch[p.lower()]
                    else:
                        si=p.lower()

                    count+=1

                    if si in numbers:
                        point=numbers.index(si)
                        if check > (point/len(numbers)):
                            mean+=check-1
                            mean/=2
                            mind.backwards([mean])
                        else:
                            mean+=1-check
                            mean/=2
                            mind.backwards([mean])
                        c_eval+=abs((point/len(numbers))-check)
                        score+=abs(check-(point/len(numbers)))
                        inputs=[0]*len(numbers)
                        inputs[point]=1
                        #check=mind.forward(inputs)[0]
                        check=mind.forward([])[0]
                        if si==' ':
                            if word_considered in word_list:
                                if word_considered in word_step and (word_eval/word_count) < 0.01:
                                    word_step[word_considered]+=1
                                elif (word_eval/word_count) < 0.01:
                                    print(word_considered)
                                    word_step[word_considered]=1
                                if word_count> 0 and (word_eval/word_count) < word_list[word_considered]:
                                    word_list[word_considered]=(word_eval/word_count)
                            elif word_count> 0:
                                if (word_eval/word_count) < 0.01:
                                    word_step[word_considered]=1
                                word_list[word_considered]=(word_eval/word_count)
                            word_eval=0
                            word_count=0
                            word_considered=''
                        else:
                            word_considered+=si
                            word_eval+=abs(check-(point/len(numbers)))
                            word_count+=1

            del page._objects
            del page._layout
        return word_step        

def testing(test):
    count=0
    word_considered=''
    wordlist={}
    #if os.path.exists(r'C:\Data\jabberrer.pkl'):
    #    GENE=open(r'C:\Data\jabberrer.pkl','rb')
    #    unpickler = pickle.Unpickler(GENE)
    #    left_mind = unpickler.load()
    #    GENE.close()
    #else:
    # specify your path of directory

    path = r"C:\Data\Novels"
    no_noise=0
    legnths={}
    # call listdir() method
    # path is a directory of which you want to list
    directories = os.listdir( path )

    notes=['\n',]
    
    file='kjv_bible_with_apocrypha.pdf'

    with pdfplumber.open(path+"\\"+file) as pdf:
        for page in pdf.pages:
            printin=page.extract_text()
            if printin !=None:
                for p in printin:
                    count+=1
                    if not p in notes:
                        sips=p.lower()
                        if sips==' ':
                            if word_considered in wordlist:
                                wordlist[word_considered]+=1
                            else:
                                no_noise+=1
                                wordlist[word_considered]=1
                            if len(word_considered) in legnths:
                                legnths[len(word_considered)]+=1
                            else:
                                legnths[len(word_considered)]=1
                            word_considered=''
                        else:
                            word_considered+=sips
            del page._objects
            del page._layout
        print(legnths)
        print(no_noise)
        print(count)
        count=0
        for word in wordlist:
            count+=wordlist[word]
        values={}
        for word in test:
            try:
                print("*"*50)
                print(word)
                print("likelihood of event")
                print(wordlist[word]/count)
                print("likelihood of AI saying this randomly")
                print(1/(61**len(word)))
                print("Times this happened")
                print(test[word])
                print("likelihood this could happen")
                print(str(((wordlist[word]/count)*(1/(61**len(word))))))
                print("likelihood this could happen in this book")
                print(str((((wordlist[word]/count)*(1/(61**len(word))))*count)*(wordlist[word]/test[word])))
                values[word]=1/(((wordlist[word]/count)*(1/(61**len(word))))*count)*(wordlist[word]/test[word])
            except:
                pass
        count=0
        for val in values:
            count+=math.log(values[val],10)
        return count

def training(number,scale):
    #if os.path.exists(r'C:\Data\jabberrer.pkl'):
    #    GENE=open(r'C:\Data\jabberrer.pkl','rb')
    #    unpickler = pickle.Unpickler(GENE)
    #    left_mind = unpickler.load()
    #    GENE.close()
    #else:
    patterns=[]
    sized=[]
    a=len(numbers)
    for i in range(scale):
        sized.append(scale)
        patterns.append((0, 0, 0, [1], [0.5]))
        a=scale
    sized.append(1)
    patterns.append((0, 0, 0, [0], [0.5]))

    left_mind=neuralnetwork(inputs=(len(numbers)+1),size=sized,rate=0.1,patterns=patterns)
    right_mind=neuralnetwork(inputs=(len(numbers)+1),size=sized,rate=0.1,patterns=patterns)


    # specify your path of directory

    path = r"C:\Data\Novels"
     
    # call listdir() method
    # path is a directory of which you want to list
    directories = os.listdir( path )
    inputs=[0]*(len(numbers)+1)
    check=left_mind.forward(np.array(inputs))
    evaluate=(1/len(numbers))

    file='kjv_bible_with_apocrypha.pdf'

    with pdfplumber.open(path+"\\"+file) as pdf:
        num_of_pages = len(pdf.pages)
        a_eval=[]
        b_eval=[]
        c_eval=0
        counts=[]
        count=0
        page_point=0
        pdf=pdf.pages
        page_ranges = len(pdf)
        scored=[]
        pages=[]
        word_champ=''
        word_best=1
        check1=0.5
        check2=0.5
        for page in pdf:
            word_eval=0
            word_count=0
            word_considered=''
            score=0
            page_point+=1
            printin=page.extract_text()
            if printin !=None:
                for p in printin:
                    if p.lower() in switch:
                        si=switch[p.lower()]
                    else:
                        si=p.lower()

                     
                    if si in numbers:

                        point=numbers.index(si)
                        inputs=[0]*len(numbers)
                        update=check
                        left_mind.backwards((check-check2)-(point/len(numbers)))
                        right_mind.backwards((point/len(numbers))-(check-check1))
                        a_eval.append(check)
                        b_eval.append(point/len(numbers))
                        c_eval+=abs(check-(point/len(numbers)))
                        score+=abs(check-(point/len(numbers)))
                        counts.append(count)
                        inputs[point]=1
                        check1=left_mind.forward(np.array(inputs))
                        check2=right_mind.forward(np.array(inputs))
                        check=check1-check2
                        if check < 0:
                            check=0
                        count+=1
                        if si==' ':
                            if word_count> 0 and (word_eval/word_count) < word_best:
                                word_best=(word_eval/word_count)
                                word_champ=word_considered
                            word_eval=0
                            word_count=0
                            word_considered=''
                        else:
                            word_considered+=si
                            word_eval+=abs(check-(point/len(numbers)))
                            word_count+=1
                scored.append(score)
                pages.append(page_point)
                del page._objects
                del page._layout
                sns.set_context("poster")
                fig, ax = plt.subplots(figsize=(12,5))
                ax2 = ax.twinx()
                ax.plot(np.array(pages), np.array(scored), color='green', marker='>')
                ax.yaxis.grid(color='lightgray', linestyle='dashed')
                plt.tight_layout()
                plt.savefig('C:\\Data\\graphs\\pages_'+str(scale)+'.png')
                plt.close()
        
        print("best word")
        print(word_champ)
        print(word_best)
        sns.set_context("poster")
        fig, ax = plt.subplots(figsize=(12,5))
        ax2 = ax.twinx()
        ax.plot(np.array(counts[-250:]), np.array(a_eval[-250:]), color='green', marker='>')
        ax2.plot(np.array(counts[-250:]), np.array(b_eval[-250:]), color='blue', marker='o')
        ax.yaxis.grid(color='lightgray', linestyle='dashed')
        plt.tight_layout()
        plt.savefig('C:\\Data\\graphs\\voices_'+str(scale)+'.png')
        plt.close()
        check=left_mind.forward(np.array(inputs))
        evaluate=(1/len(numbers))
        inputs=[0]*len(numbers)
        thing=''
        for i in range(1000):
            int(check/evaluate)
            point=numbers.index(numbers[int(check/evaluate)])
            thing+=numbers[point]
            inputs=[0]*len(numbers)
            #left_mind.backwards(check,(point/len(numbers)))
            inputs[point]=1
            check=left_mind.forward(np.array(inputs))#
        print(thing)
                 
        GENE=open(r'C:\Data\jabberrer.pkl','wb')
        #pickle.dump(left_mind,GENE)
        thing=''
        return c_eval
    



if not os.path.exists(r'C:\Data\graphs\pride_prejudice_3d_line1_siameseV80.pkl'):
    list1=[]
    list2=[]
    list3=[]
    GENE = open(r'C:\Data\graphs\pride_prejudice_3d_line1_siameseV80.pkl', 'wb')
    pickle.dump(list1,GENE)
    GENE.close()
    GENE = open(r'C:\Data\graphs\pride_prejudice_3d_line2_siameseV80.pkl', 'wb')
    pickle.dump(list2,GENE)
    GENE.close()
    GENE = open(r'C:\Data\graphs\pride_prejudice_3d_line3_siameseV80.pkl', 'wb')
    pickle.dump(list3,GENE)
    GENE.close()
else:
    GENE=open(r'C:\Data\graphs\pride_prejudice_3d_line1_siameseV80.pkl','rb')
    unpickler = pickle.Unpickler(GENE)
    list1 = unpickler.load()
    GENE.close()
    GENE=open(r'C:\Data\graphs\pride_prejudice_3d_line2_siameseV80.pkl','rb')
    unpickler = pickle.Unpickler(GENE)
    list2 = unpickler.load()
    GENE.close()
    GENE=open(r'C:\Data\graphs\pride_prejudice_3d_line3_siameseV80.pkl','rb')
    unpickler = pickle.Unpickler(GENE)
    list3 = unpickler.load()
    GENE.close()

b=2
c=2
support=-1

switch={}
numbers=[' ', '!', '"','&', "'", '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', '_','!','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '–', '—', '‘', '’', '“', '”', '…']
switch['\n']=' '
switch['á']='a'
switch['é']='e'
switch['í']='i'
switch['ú']='u'
switch['ý']='y'
switch['č']='c'
switch['ě']='e'
switch['ň']='n'
switch['ř']='r'
switch['š']='s'
switch['ů']='u'
switch['ž']='z'
switch['×']='x'
switch['ß']='b'
switch['à']='a'
switch['á']='a'
switch['ã']='a'
switch['ä']='a'
switch['å']='a'
switch['æ']='a'
switch['ç']='c'
switch['è']='e'
switch['é']='e'
switch['ê']='e'
switch['ë']='e'
switch['í']='i'
switch['í']='i'
switch['î']='i'
switch['î']='i'
switch['ï']='i'

#test=True
#while test==True:
#    test=False
#    for i in range(len(list1)):
#        if list1[i]==b and list2[i]==c:
#            test=True
#            c+=1

def recursion_build(brain,cortex,mate,size,current=0,maxi=2):
    
    row=brain.honey_comb(neurones=cortex,pattern=2,value=1)
    row=brain.honey_comb(neurones=row,pattern=2,value=1)
    
    brain.add_neurones(row,to_add=[mate],value=1,split=False)
    current+=1
    if current < maxi:
        for neurone in row:
            cortex=brain.divergence(neurones=[neurone])
            for part in range(len(cortex)):
                row=brain.divergence([cortex[part]],steps=size,value=0)
                recursion_build(brain,row,cortex[part-1],size,current=current,maxi=maxi)

while True:

    test=True
    while test==True:
        test=False
        for i in range(len(list1)):
            if list1[i]==b and list2[i]==c:
                test=True
                c+=1
    brain=mind()
    #for i in range(len(numbers)):
        #brain.add_input()
    brain.add_output()
    cortex=brain.divergence()
    for part in range(len(cortex)):
        row=brain.divergence([cortex[part]],steps=b,value=0)
        recursion_build(brain,row,cortex[part-1],size=b,current=0,maxi=c)
        
        
    test=train_once(numbers,brain)

    GENE=open(r'C:\Data\graphs\pride_prejudice_3d_line1_siameseV80.pkl','rb')
    unpickler = pickle.Unpickler(GENE)
    list1 = unpickler.load()
    GENE.close()
    GENE=open(r'C:\Data\graphs\pride_prejudice_3d_line2_siameseV80.pkl','rb')
    unpickler = pickle.Unpickler(GENE)
    list2 = unpickler.load()
    GENE.close()
    GENE=open(r'C:\Data\graphs\pride_prejudice_3d_line3_siameseV80.pkl','rb')
    unpickler = pickle.Unpickler(GENE)
    list3 = unpickler.load()
    GENE.close()

    test=testing(test)
    list1.append(b)
    list2.append(c)
    list3.append(test)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(np.array(list1), np.array(list2), np.array(list3), color='blue')
    plt.savefig(r'C:\\Data\\graphs\\entropy_calcs_3d_siameseV80.png')
    plt.close()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(np.array(list3),np.array(list1), np.array(list2), color='blue')
    plt.savefig('C:\\Data\\graphs\\entropy_calcs_3d_siamese_onsideV80.png')
    plt.close()
    GENE = open(r'C:\Data\graphs\pride_prejudice_3d_line1_siameseV80.pkl', 'wb')
    pickle.dump(list1,GENE)
    GENE.close()
    GENE = open(r'C:\Data\graphs\pride_prejudice_3d_line2_siameseV80.pkl', 'wb')
    pickle.dump(list2,GENE)
    GENE.close()
    GENE = open(r'C:\Data\graphs\pride_prejudice_3d_line3_siameseV80.pkl', 'wb')
    pickle.dump(list3,GENE)
    GENE.close()

    if test > support:
        support=test
        c+=1
    else:
        support=0
        b+=1
        c=1
