import bpy
import time
import math
import csv
import pandas
import random

scn = bpy.context.scene

camera1 = bpy.data.objects['Camera1']
camera2 = bpy.data.objects['Camera2']

#plane = bpy.data.objects['Plane']

#lamp = bpy.data.objects['Lamp']
lamp2 = bpy.data.objects['Lamp2']
#bpy.data.images['HDR_111_Parking_Lot_2_Bg.jpg'].name = 'Background'
#texture

barbell = bpy.data.objects['Barbell'] 
disk01 = bpy.data.objects['Disk01']
disk02 = bpy.data.objects['Disk02']
disk03 = bpy.data.objects['Disk03']
disk04 = bpy.data.objects['Disk04']
disk05 = bpy.data.objects['Disk05']
disk06 = bpy.data.objects['Disk06']
disk07 = bpy.data.objects['Disk07']
disk08 = bpy.data.objects['Disk08']
disk09 = bpy.data.objects['Disk09']
disk10 = bpy.data.objects['Disk10']
disk11 = bpy.data.objects['Disk11']
disk12 = bpy.data.objects['Disk12']
disk13 = bpy.data.objects['Disk13']
disk14 = bpy.data.objects['Disk14']
disk15 = bpy.data.objects['Disk15']
disk16 = bpy.data.objects['Disk16']
#material setup
md01 = bpy.data.materials['disk01'].node_tree.nodes.get("Glossy BSDF")
md02 = bpy.data.materials['disk02'].node_tree.nodes.get("Glossy BSDF")
md03 = bpy.data.materials['disk03'].node_tree.nodes.get("Glossy BSDF")
md04 = bpy.data.materials['disk04'].node_tree.nodes.get("Glossy BSDF")
md05 = bpy.data.materials['disk05'].node_tree.nodes.get("Glossy BSDF")
md06 = bpy.data.materials['disk06'].node_tree.nodes.get("Glossy BSDF")
md07 = bpy.data.materials['disk07'].node_tree.nodes.get("Glossy BSDF")
md08 = bpy.data.materials['disk08'].node_tree.nodes.get("Glossy BSDF")
md09 = bpy.data.materials['disk09'].node_tree.nodes.get("Glossy BSDF")
md10 = bpy.data.materials['disk10'].node_tree.nodes.get("Glossy BSDF")
md11 = bpy.data.materials['disk11'].node_tree.nodes.get("Glossy BSDF")
md12 = bpy.data.materials['disk12'].node_tree.nodes.get("Glossy BSDF")
md13 = bpy.data.materials['disk13'].node_tree.nodes.get("Glossy BSDF")
md14 = bpy.data.materials['disk14'].node_tree.nodes.get("Glossy BSDF")
md15 = bpy.data.materials['disk15'].node_tree.nodes.get("Glossy BSDF")
md16 = bpy.data.materials['disk16'].node_tree.nodes.get("Glossy BSDF")
mb = bpy.data.materials['barbell'].node_tree.nodes.get("Glossy BSDF")

#Distribution (for glossy bsdf materials)
distribution_list = ["SHARP", "GGX"]
#WORLD MIX OPTIONS
type_list = ["ADD","DIVIDE","MULTIPLY","MIX","SCREEN"]
#red, green, blue, yellow, black, gray(barbell).. (normalized)
plate_colors = ([0.62745098, 0.21176471, 0.20392157,1],[0.1372, 0.6901, 0.1803,1],[0.11372549, 0.36470588, 0.67843137,1],[0.92156863, 0.75294118, 0.10196078,1],[0.1764, 0.1764, 0.1764,1],[0.706, 0.706, 0.706, 1])
# normalized red: 1, 0, 0 
# normalized green: 0, 1, 0
# normalized blue: 0, 0, 1
# normalized yellow: 1, 1, 0
# normalized black: 0.011, 0.011, 0.011
# gray

def setup():   
    # shadder of plane
    #plane.active_material.diffuse_shader = 'FRESNEL'
    scn.cycles.device = 'GPU'
    
    # get the nodes
    
    md01.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md01.inputs[1].default_value = 0.5

    md02.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md02.inputs[1].default_value = 0.5
   
    md03.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md03.inputs[1].default_value = 0.5

    md04.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md04.inputs[1].default_value = 0.5
  
    md05.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md05.inputs[1].default_value = 0.5
 
    md06.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md06.inputs[1].default_value = 0.5
    
    md07.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md07.inputs[1].default_value = 0.5
    
    md08.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md08.inputs[1].default_value = 0.5
  
    md09.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md09.inputs[1].default_value = 0.5
      
    md10.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md10.inputs[1].default_value = 0.5
  
    md11.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md11.inputs[1].default_value = 0.5
     
    md12.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md12.inputs[1].default_value = 0.5
   
    md13.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md13.inputs[1].default_value = 0.5
      
    md14.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md14.inputs[1].default_value = 0.5
    
    md15.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md15.inputs[1].default_value = 0.5
     
    md16.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    md16.inputs[1].default_value = 0.5
 
    mb.inputs[0].default_value = (0.706, 0.706, 0.706, 1)
    mb.inputs[1].default_value = 0.5
    
    # energy of lamps
    #lamp.data.energy = 0.7
    #lamp2.data.energy = 0.7
    # render dimensions/quality
    bpy.ops.script.python_file_run(filepath="C:\\Program Files\\Blender Foundation\\Blender\\2.79\\scripts\\presets\\render\\DVCPRO_HD_1080p.py")

def render_and_save(frame):
    bpy.context.scene.render.filepath = "//images/"+str(frame)+".png"
    bpy.context.scene.render.resolution_x = 800 #perhaps set resolution in code
    bpy.context.scene.render.resolution_y = 600
    bpy.ops.render.render(use_viewport = True, write_still=True)

step = 0
locations = []
setup()
#scn.frame_end + 1
for f in range(0, 2, scn.frame_step):
    scn.frame_set(f)
    scn.cycles.device = 'GPU'
    # random floor color
#    plane.active_material.diffuse_intensity = random.random()/0.3
#    plane.active_material.diffuse_color = ((random.random()+1)/2,(random.random()+1)/2,(random.random()+1)/2)
#    plane.active_material.roughness = random.random()/3.14 #max value is 3.14
#   
            #    # random lamp positions, lamp 1
            #    lamp.location[0] = random.randint(0,5)
            #    lamp.location[1] = random.randint(0,5)
            #    lamp.location[2] = random.randint(4,5)
    # random lamp positions, lamp 2
#    # sun for HDR_111_Parking_Lot_2_Bg.jpg
#    lamp2.data.type = 'SUN'
#    lamp2.location[0] = 20
#    lamp2.location[1] = 2
#    lamp2.location[2] = 10
    
    #WORLD setup
    bpy.data.worlds['World'].node_tree.nodes['Mix'].blend_type = "MIX"#type_list[random.randint(0,4)]
    bpy.data.worlds['World'].node_tree.nodes['Mix.001'].blend_type = "ADD"#type_list[random.randint(0,4)]
    bpy.data.worlds['World'].node_tree.nodes['Mix.002'].blend_type = "MIX"#type_list[random.randint(0,4)]
    bpy.data.worlds['World'].node_tree.nodes['Mix'].inputs[0].default_value = random.random()
    bpy.data.worlds['World'].node_tree.nodes['Mix.001'].inputs[0].default_value = random.random()
    bpy.data.worlds['World'].node_tree.nodes['Mix.002'].inputs[0].default_value = random.random()
    bpy.data.worlds['World'].node_tree.nodes['Background'].inputs['Strength'].default_value = random.uniform(0.3,0.6)
    
    # random disc colors (they have to be paired)
    md01.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md01.inputs[1].default_value = random.uniform(0.3,0.6)
    
    md02.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md02.inputs[1].default_value = random.uniform(0.3,0.6)
   
    md03.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md03.inputs[1].default_value = random.uniform(0.3,0.6)
   
    md04.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md04.inputs[1].default_value = random.uniform(0.3,0.6)
 
    md05.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md05.inputs[1].default_value = random.uniform(0.3,0.6)

    md06.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md06.inputs[1].default_value = random.uniform(0.3,0.6)
  
    md07.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md07.inputs[1].default_value = random.uniform(0.3,0.6)
   
    md08.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md08.inputs[1].default_value = random.uniform(0.3,0.6)

    md09.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md09.inputs[1].default_value = random.uniform(0.3,0.6)
   
    md10.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md10.inputs[1].default_value = random.uniform(0.3,0.6)
   
    md11.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md11.inputs[1].default_value = random.uniform(0.3,0.6)
 
    md12.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md12.inputs[1].default_value = random.uniform(0.3,0.6)

    md13.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md13.inputs[1].default_value = random.uniform(0.3,0.6)

    md14.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md14.inputs[1].default_value = random.uniform(0.3,0.6)
   
    md15.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md15.inputs[1].default_value = random.uniform(0.3,0.6)
   
    md16.inputs[0].default_value = plate_colors[random.randint(0,5)]
    md16.inputs[1].default_value = random.uniform(0.3,0.6)
 
    mb.distribution = random.choice(distribution_list)
    mb.inputs[0].default_value = plate_colors[random.randint(0,5)]
    mb.inputs[1].default_value = random.uniform(4.5,8.55)

    barbell.location[2] = random.random()*2
    bpy.context.scene.render.use_motion_blur = bool(random.getrandbits(1))
    
    #hide/show disks
    disk03.hide_render = bool(random.getrandbits(1))
    disk04.hide_render = disk03.hide_render
    if disk03.hide_render:
        disk05.hide_render = True
        disk06.hide_render = True
        disk07.hide_render = True
        disk08.hide_render = True
        disk09.hide_render = True
        disk10.hide_render = True
        disk11.hide_render = True
        disk12.hide_render = True
        disk13.hide_render = True
        disk14.hide_render = True
        disk15.hide_render = True
        disk16.hide_render = True
    else:
        disk05.hide_render = bool(random.getrandbits(1))
        disk06.hide_render = disk05.hide_render
        if disk05.hide_render:
            disk07.hide_render = True
            disk08.hide_render = True
            disk09.hide_render = True
            disk10.hide_render = True
            disk11.hide_render = True
            disk12.hide_render = True
            disk13.hide_render = True
            disk14.hide_render = True
            disk15.hide_render = True
            disk16.hide_render = True
        else:
            disk07.hide_render = bool(random.getrandbits(1))
            disk08.hide_render = disk07.hide_render
            if disk07.hide_render:
                disk09.hide_render = True
                disk10.hide_render = True
                disk11.hide_render = True
                disk12.hide_render = True
                disk13.hide_render = True
                disk14.hide_render = True
                disk15.hide_render = True
                disk16.hide_render = True
            else:
                disk09.hide_render = bool(random.getrandbits(1))
                disk10.hide_render = disk09.hide_render
                if disk09.hide_render:
                    disk11.hide_render = True
                    disk12.hide_render = True
                    disk13.hide_render = True
                    disk14.hide_render = True
                    disk15.hide_render = True
                    disk16.hide_render = True
                else:
                    disk11.hide_render = bool(random.getrandbits(1))
                    disk12.hide_render = disk11.hide_render
                    if disk11.hide_render:
                        disk13.hide_render = True
                        disk14.hide_render = True
                        disk15.hide_render = True
                        disk16.hide_render = True
                    else:
                        disk13.hide_render = bool(random.getrandbits(1))
                        disk14.hide_render = disk13.hide_render
                        if disk13.hide_render:
                            disk15.hide_render = True
                            disk16.hide_render = True
                        else:
                            disk15.hide_render = bool(random.getrandbits(1))
                            disk16.hide_render = disk15.hide_render
                            
    #append camera 1 parameters
    locations.append([camera1.location[0],camera1.location[1],camera1.location[2],
    camera1.rotation_euler[0],camera1.rotation_euler[1],camera1.rotation_euler[2]])
    #append camera 2 parameters
    locations.append([camera2.location[0],camera2.location[1],camera2.location[2],
    camera2.rotation_euler[0],camera2.rotation_euler[1],camera2.rotation_euler[2]]) 
    
    currentCameraObj = bpy.data.objects['Camera1']
    scn.camera = currentCameraObj
    
    render_and_save(step)
    step += 1
    
    currentCameraObj = bpy.data.objects['Camera2']
    scn.camera = currentCameraObj
    
    render_and_save(step)
    step += 1

df = pandas.DataFrame(data=locations,columns=['x_l','y_l','z_l','x_r','y_r','z_r'])
df.to_csv('C:/Users/ruben/Documents/Github/bar-detection/file2.csv')