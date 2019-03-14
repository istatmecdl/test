'''
Created on 12/12/2018
Modified on 26/02/2018

@author: Francesco Pugliese
'''

import pdb

class Postprocessing:

    @staticmethod
    def labels_to_eurosat_classes_converter(class_number):
        if class_number == 0:
            class_txt='Annual Crop'
        elif class_number == 1:
            class_txt='Forest'
        elif class_number == 2:
            class_txt='Herbaceous Vegetation'
        elif class_number == 3:
            class_txt='Highway'
        elif class_number == 4:
            class_txt='Industrial'
        elif class_number == 5:
            class_txt='Pasture'
        elif class_number == 6:
            class_txt='Permanent Crop'
        elif class_number == 7:
            class_txt='Residential'
        elif class_number == 8:
            class_txt='River'
        elif class_number == 9:
            class_txt='Sea Lake'
       
        return class_txt
        
    @staticmethod
    def eurosat_labels_counting(coords_list, classes_list, language):
        classes_array_list = []
                
        # Count of predicted classes
        raccolto_annuale = 0
        foresta = 0
        vegetazione_erbacea = 0
        strada = 0
        industriale = 0
        pascolo = 0
        coltura_permanente = 0
        residenziale = 0
        fiume = 0
        lago = 0
        
        # Draw split image in coords_list
        for i in range(len(classes_list)):

            if classes_list[i]== [0]:
                class_txt='raccolto annuale'
                raccolto_annuale +=1
                colors_boxes = 'khaki'
                
            elif classes_list[i]== [1]:
                class_txt='foresta'
                foresta += 1
                colors_boxes = 'green'

            elif classes_list[i]== [2]:
                class_txt='vegetazione erbacea'
                vegetazione_erbacea += 1
                colors_boxes = 'yellowgreen'

            elif classes_list[i]== [3]:
                class_txt='strada'
                strada += 1
                colors_boxes = 'grey'

            elif classes_list[i]== [4]:
                class_txt='industriale'
                industriale += 1
                colors_boxes = 'peru'

            elif classes_list[i]== [5]:
                class_txt='pascolo'
                pascolo += 1
                colors_boxes = 'whitesmoke'
                
            elif classes_list[i]== [6]:
                class_txt='coltura permanente'
                coltura_permanente += 1
                colors_boxes = 'mediumseagreen'
               
            elif classes_list[i]== [7]:
                class_txt='residenziale'
                residenziale += 1
                colors_boxes = 'beige'

            elif classes_list[i]== [8]:
                class_txt='fiume'
                fiume += 1
                colors_boxes = 'aqua'

            elif classes_list[i]== [9]:
                class_txt='lago'
                lago += 1
                colors_boxes = 'cyan'
            
            classes_array_list.append([coords_list[i][0],coords_list[i][2],coords_list[i][1],coords_list[i][3],class_txt])

            #print("\n%i, %i, %i, %i, %s " % (coords_list[i][0],coords_list[i][2],coords_list[i][1],coords_list[i][3],class_txt))
            '''
            # display the predictions to our screen
            if coords_list[i][0] > coords_list[i-1][0]:
                #print("\n(%i, %i) - (%i, %i) : %s , %s" % (coords_list[i][0],coords_list[i][2],coords_list[i][1],coords_list[i][3],class_txt, colors_boxes))
                #print("\n%i %i %s %s" % (coords_list[i][2],coords_list[i][0], class_txt))

            else:
                #print("(%i, %i) - (%i, %i) : %s , %s" % (coords_list[i][0],coords_list[i][2],coords_list[i][1],coords_list[i][3],class_txt, colors_boxes))
                print("%i %i %s %s" % (coords_list[i][2],coords_list[i][0], class_txt))
            '''

        countings = [raccolto_annuale, foresta, vegetazione_erbacea, strada, industriale, pascolo, coltura_permanente, residenziale, fiume, lago]
        
        return [classes_array_list, countings]
            
    @staticmethod
    def lucas_labels_counting(coords_list, classes_list, language):
        classes_array_list = []

        # Count predicted classes
        crop_land = 0
        wood_land = 0
        grass_land = 0
        artificial_land = 0
        water_areas = 0
        
        # Draw split image in coords_list
        for i in range(len(classes_list)):

            if classes_list[i]== [0]:
                class_txt='Crop Land'
                crop_land +=1
                colors_boxes = 'khaki'
                
            elif classes_list[i]== [1]:
                class_txt='Wood Land'
                wood_land += 1
                colors_boxes = 'green'

            elif classes_list[i]== [2]:
                class_txt='Grass Land'
                grass_land += 1
                colors_boxes = 'yellowgreen'

            elif classes_list[i]== [3]:
                class_txt='Artificial Land'
                artificial_land += 1
                colors_boxes = 'grey'

            elif classes_list[i]== [4]:
                class_txt='Aritificial Land'
                artificial_land += 1
                colors_boxes = 'grey'

            elif classes_list[i]== [5]:
                class_txt='Crop Land'
                crop_land += 1
                colors_boxes = 'khaki'
                
            elif classes_list[i]== [6]:
                class_txt='Crop Land'
                crop_land += 1
                colors_boxes = 'khaki'
               
            elif classes_list[i]== [7]:
                class_txt='Artificial Land'
                artificial_land += 1
                colors_boxes = 'grey'

            elif classes_list[i]== [8]:
                class_txt='Water Areas'
                water_areas += 1
                colors_boxes = 'aqua'

            elif classes_list[i]== [9]:
                class_txt='Water Areas'
                water_areas += 1
                colors_boxes = 'aqua'
                
            classes_array_list.append([coords_list[i][0],coords_list[i][2],coords_list[i][1],coords_list[i][3],class_txt])

            #print("\n%i, %i, %i, %i, %s " % (coords_list[i][0],coords_list[i][2],coords_list[i][1],coords_list[i][3],class_txt))
            '''
            # display the predictions to our screen
            if coords_list[i][0] > coords_list[i-1][0]:
                #print("\n(%i, %i) - (%i, %i) : %s , %s" % (coords_list[i][0],coords_list[i][2],coords_list[i][1],coords_list[i][3],class_txt, colors_boxes))
                #print("\n%i %i %s %s" % (coords_list[i][2],coords_list[i][0], class_txt))

            else:
                #print("(%i, %i) - (%i, %i) : %s , %s" % (coords_list[i][0],coords_list[i][2],coords_list[i][1],coords_list[i][3],class_txt, colors_boxes))
                print("%i %i %s %s" % (coords_list[i][2],coords_list[i][0], class_txt))
            '''

        countings = [crop_land, wood_land, grass_land, artificial_land, water_areas]
        
        return [classes_array_list, countings]
        
    @staticmethod
    def eurosat_statistics_compute(countings, coords_list, language):
        # Inizialize arrays of the statistics
        fracs = []
        labels = []
        explode = []
        colors = []
    
        if countings[0] != 0:
            rac = (countings[0]/len(coords_list))*100
            print('raccolto annuale:',rac,'%')
            fracs.append(rac)
            labels.append('raccolto annuale')
            explode.append(0.1)
            colors.append('khaki')
        if countings[1] != 0: 
            fore = (countings[1]/len(coords_list))*100
            print('foresta:',fore,'%')
            fracs.append(fore)
            labels.append('foresta')
            explode.append(0.1)
            colors.append('green')
        if countings[2] != 0: 
            veg = (countings[2]/len(coords_list))*100
            print('vegetazione erbacea:',veg,'%')
            fracs.append(veg)
            labels.append('veg. erbacea')
            explode.append(0.1)
            colors.append('yellowgreen')
        if countings[3] != 0:
            strada = (countings[3]/len(coords_list))*100
            print('strada:',strada,'%')
            fracs.append(strada)
            labels.append('strada')
            explode.append(0.1)
            colors.append('grey')
        if countings[4] != 0:
            ind = (countings[4]/len(coords_list))*100
            print('industriale:' ,ind,'%')
            fracs.append(ind)
            labels.append('industriale')
            explode.append(0.1)
            colors.append('peru')
        if countings[5] != 0:
            pas = (countings[5]/len(coords_list))*100
            print('pascolo:',pas,'%')
            fracs.append(pas)
            labels.append('pascolo')
            explode.append(0.1)
            colors.append('whitesmoke')
        if countings[6] != 0:
            col = (countings[6]/len(coords_list))*100
            print('coltura permanente:',col,'%')
            fracs.append(col)
            labels.append('coltura permanente')
            explode.append(0.1)
            colors.append('mediumseagreen')
        if countings[7] != 0:
            res = (countings[7]/len(coords_list))*100
            print('residenziale:',res,'%')
            fracs.append(res)
            labels.append('residenziale')
            explode.append(0.1)
            colors.append('beige')
        if countings[8] != 0: 
            fiu = (countings[8]/len(coords_list))*100
            print('fiume:',fiu,'%')
            fracs.append(fiu)
            labels.append('fiume')
            explode.append(0.1)
            colors.append('aqua')
        if countings[9] != 0: 
            la = (countings[9]/len(coords_list))*100
            print('lago:',la,'%')
            fracs.append(la)
            labels.append('lago')
            explode.append(0.1)
            colors.append('cyan')

        return [fracs, labels, explode, colors]

    @staticmethod
    def lucas_statistics_compute(countings, coords_list, language):
        # Inizialize arrays of the statistics
        fracs = []
        labels = []
        explode = []
        colors = []

        if countings[0] != 0:                           # Crop Land
            crop = (countings[0]/len(coords_list))*100
            print('Cropland:',crop,'%')
            fracs.append(crop)
            labels.append('Cropland')
            explode.append(0.1)
            colors.append('khaki')
        if countings[1] != 0:                           # Wood Land
            wood = (countings[1]/len(coords_list))*100
            print('Woodland:',wood,'%')
            fracs.append(wood)
            labels.append('Woodland')
            explode.append(0.1)
            colors.append('green')
        if countings[2] != 0:                           # Grass Land
            grass = (countings[2]/len(coords_list))*100
            print('Grassland:',grass,'%')
            fracs.append(grass)
            labels.append('Grassland')
            explode.append(0.1)
            colors.append('yellowgreen')
        if countings[3] != 0:                           # Artificial Land
            art = (countings[3]/len(coords_list))*100
            print('Artificial Land:',art,'%')
            fracs.append(art)
            labels.append('Artificial Land')
            explode.append(0.1)
            colors.append('grey')
        if countings[4] != 0:                           # Water Areas
            water = (countings[4]/len(coords_list))*100
            print('Water Areas:' ,water,'%')
            fracs.append(water)
            labels.append('Water Areas')
            explode.append(0.1)
            colors.append('aqua')
    
        return [fracs, labels, explode, colors]
        
    @staticmethod
    def get_plot_classes_title(parameters):
        if parameters.classification_type == "EuroSAT":  
            classes_title = "EuroSat Classification \n\n"
            ground_truth_title = "EuroSAT Classification   n\n Ground Truth"
        elif parameters.classification_type == "Lucas": 
            classes_title = "Lucas Classification   \n\n"
            ground_truth_title = "Lucas Classification   \n\n Ground Truth"
        
        if parameters.quantization == True: 
            quantization_title = classes_title + "Quantification by different samplings"
        else: 
            quantization_title = None

        if parameters.rotate_tiles == True: 
            rotation = "On"
        else: 
            rotation = "Off"
        classes_title = classes_title + " Stride: "+parameters.stride.__str__()+", Rotation: "+rotation
        if parameters.rotate_tiles == True: 
            classes_title = classes_title + ", Rotation type: "
            if parameters.random_rotations == True:
                classes_title = classes_title + "Random"
            else: 
                classes_title = classes_title + "180"
        
        return [classes_title, ground_truth_title, quantization_title]
     