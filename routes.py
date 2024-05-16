from flask import render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import *
from app import app
from PIL import Image
import os
from python_code.main import cut_image, getSquareOfFace
# from run import getDevice, getMtcnn, getResnet
from python_code.ProcessImageAndSave import ProcesarImagenYGuardar, id_face, search_person, id_face_compare_two_faces,points_2d
from flask import send_from_directory as send_from_directory_flask
from python_code.make_folders import makeFoldersFor
from python_code.pdf_maker import dat
from python_code.pdf_makerV2 import dat as datV2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from zipfile import ZipFile
from flask import Flask, request, render_template
from python_code.put_watermark import putWatermark
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
from python_code.ZipImages import *
from python_code.helpers.HandleInputs import HandleAnInput, default_response, ValidIdentifiers
from python_code.configs_basics import folder_to_search
from python_code.pdf_makerV2 import to_data_url
from python_code.Manage_DB import get_user_DB, register_user_photo
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, create_refresh_token
from python_code.helpers.FetchUserDataAPI import getInfoFromAPI
from python_code.helpers.HandleDataUser import HandleDataUser
from datetime import timedelta
from datetime import datetime
from python_code.watermark import watermark, get_latest_folder, get_latest_f
from functools import wraps
import jwt as jwtf

import logging


logging.basicConfig(filename='registro.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s')

@app.before_request
def start_request():
    request.start_time = datetime.now()
    # app.logger.info(f"Request: {str(request.blueprint)}")
    app.logger.info('Inicio de la solicitud: %s %s %s %s %s', 
                    request.remote_addr, request.method, request.scheme, request.full_path, str(request.form))

@app.after_request
def log_request(response):
    # Obtenemos el tiempo de inicio de la solicitud
    start_time = request.start_time

    # Calculamos la duración de la solicitud
    request_duration = datetime.now() - start_time

    # Registramos los detalles de la solicitud
    app.logger.info('Tiempo de solicitud: %s, Direccion IP: %s, Metodo: %s, Ruta: %s, Duracion: %s, Codigo de estado: %s', 
                    start_time, request.remote_addr, request.method, request.path, request_duration, response.status_code)

    # app.logger.info(f"Response: {str(response)}")
    return response

bcrypt = Bcrypt(app)
jwt = JWTManager(app)
device = torch.device('cpu')
# model.to(device)
print('Running on device: {}'.format(device))
mtcnn = MTCNN(
        select_largest=True,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        post_process=False,
        image_size=700, 
        device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def getDevice():

    '''
    Obtiene el dispositivo en el que se está ejecutando el modelo

    returns: El dispositivo de ejecución del modelo
    '''
    return device

def getMtcnn():

    '''
    Obtiene el detector de caras MTCNN utilizado para la detección de caras

    returns: El detector de caras MTCNN
    '''
    return mtcnn

def getResnet():

    '''
    Obtiene el modelo de reconocimiento facial ResNet utilizado para la comparación de caras

    returns: El modelo de reconocimiento facial ResNet
    '''
    return resnet


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def validar_fecha(cadena):
    # Definimos el patrón de la expresión regular para el formato de fecha "YYYY-MM-DD"
    patron = r'^\d{4}-\d{2}-\d{2}$'
    
    # Comprobamos si la cadena coincide con el patrón
    if re.match(patron, cadena):
        return True
    else:
        return False


@app.route('/')
def index():
    '''
    Ruta principal de la aplicación Flask

    returns: Un mensaje ¡Hola, Flask!
    '''
    return '¡Hola, Flask!'

@app.route('/api/pdf/<path:name>', methods=["GET"])
@jwt_required()
def obtener_pdf(name):
    '''
    Obtiene un archivo PDF

    Parameters:

    name    - El nombre del archivo PDF

    returns: La respuesta que contiene el archivo PDF
    '''
    # return send_from_directory_flask("../all_pdf", path=name, as_attachment=True)
    return send_from_directory_flask("../all_pdf", path=name)
    # return send_from_directory_flask("../processed_photos", path=name)

@app.route('/api/fotosv2/<path:name>')
@jwt_required()
def obtener_fotos(name):
    '''
    Obtiene una foto

    Parameters:
    name    - El nombre del archivo de la foto

    returns: La respuesta que contiene la foto
    '''
    folder = "../" + folder_to_search 
    return send_from_directory_flask(folder, path=name)
    # return send_from_directory_flask("../processed_photos", path=name)

@app.route('/api/fotos_V3/<path:name>')
@jwt_required()
def obtener_foto_bloques(name):
    # token = request.args.get('token')

    # if not token:
    #     return jsonify({'error': 'Token no proporcionado'}), 401

    # try:
    #     # Verifica el token manualmente
    #     payload = jwt._decode_key_callback(token)
    #     current_user = payload['sub']  # 'sub' es el campo por defecto que contiene la identidad del usuario en el token JWT
    # except Exception as e:
    #     print(str(e))
    #     return jsonify({'error': 'Token inválido'}), 401
    return send_from_directory_flask("../fotitos", path=name)

@app.route('/api/zipfiles/<path:name>')
@jwt_required()
def obtener_zip(name):
    return send_from_directory_flask("../all_zip", path=name, as_attachment=True)

@app.route('/api/fotos/<path:name>')
@jwt_required()
def obtener_foto(name):
    '''
    Obtiene una foto

    Parameters:
    name    - El nombre del archivo de la foto

    returns: La respuesta que contiene la foto
    '''
    # Construye la ruta completa al directorio de carga y al archivo
    # print(name)
    # directorio_carga = app.config['UPLOAD_FOLDER']
    # ruta_completa = os.path.join(app.static_folder, filename)
    
    # uploads = os.path.join("/cosa")
    # print(uploads)
    # Envía el archivo desde el directorio de carga
    return send_from_directory_flask("../Faces", path=name)

@app.route('/api/getUrlPhoto-A', methods=['POST'])
@jwt_required()
def getUrlPhoto_A():
    '''
    Obtiene las URLs de las fotos

    returns:  La respuesta que contiene las URLs de las fotos
    '''
    response_data = {
        'status': 'success',
        'message': 'Fotos cargadas correctamente',
    }
    dni_statusFN, dni, dni_code = HandleAnInput(request, "dni", "dni")
    if not dni_statusFN:
        return jsonify(dni), dni_code
    
    photos_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(dni), "redimensionada")  # Directorio donde se guardan las fotos
    if not os.path.exists(photos_dir):
        response_data['message'] = "No se encontro el dni"
        response_data['status'] = "error"
        return jsonify(response_data), 500
    photo_urls = []  # Lista para almacenar las URLs de las fotos

    # Recorrer los archivos en el directorio de fotos y generar las URLs
    for filename in os.listdir(photos_dir):
        if os.path.isfile(os.path.join(photos_dir, filename)):
            photo_urls.append("/api/fotos/" + str(dni) + "/redimensionada/" + str(filename))
    response_data['photos'] = photo_urls
    

    return jsonify(response_data), 200



@app.route('/api/tomar_fotos', methods=['GET', 'POST'])
@jwt_required()
def upload():

    '''
    Maneja la carga de fotos y realiza el recorte, redimensión y cálculo de embeddings

    returns: La respuesta que indica el estado de la carga y el procesamiento de las fotos
    '''
    # response_data = {'status': 'error', 'message': 'Ocurrió un error durante la carga del archivo.', 'statusUser': '0'}
    response_data = default_response
    try:
        if request.method == 'POST':
            dni_statusFN, dni, dni_code = HandleAnInput(request, "dni", "dni")
            if not dni_statusFN:
                return jsonify(dni), dni_code # Message and HTTP code
            
            file_statusFN, file, file_code = HandleAnInput(request, "image", "file")
            if not file_statusFN:
                return jsonify(file), file_code # Message and HTTP code
        
            codes_statusFN, codes_data, codes_code = HandleAnInput(request, "codes_school", "codigos")
            if not codes_statusFN:
                # default_response["message"] = codes_data
                return jsonify(codes_data), codes_code
            array_codigos = codes_data.split(',')
            # if not img_statusFN:
            #     return jsonify(img_data), img_code
            # return jsonify(img_data.filename)
            # date_statusFN, date_data, date_code = HandleAnInput(request, "date", "folder_paths")
            # if not date_statusFN:
                # return jsonify(date_data), date_code
            # return date_data
            #status_D, data, num_schools = HandleDataUser(getInfoFromAPI(dni))
            #if not status_D:
            #    return jsonify(data), 400
        

            filename = secure_filename(file.filename)
            filename = str(datetime.now().strftime("%Y-%m-%d-%H-%M")) + "-" + filename
            
            # path_original = os.path.join(app.config['UPLOAD_FOLDER'], dni, "original", filename)
            path_filename = os.path.join(app.config['UPLOAD_FOLDER'], dni, "original", filename)

            if(makeFoldersFor(dni)):
                file.save(path_filename)
                # file.save(path_original)
                # path_renamed = os.path.join(app.config['UPLOAD_FOLDER'], dni, "fotos", f"{dni}.jpg")
                # os.rename(path_filename, path_renamed)
            else:
                response_data['message'] = 'Hubo un error al crear las carpetas'
                return jsonify(response_data), 400
            
                # print("aqui")
            # response_data['status'] = 'success'
            # response_data['message'] = 'Archivo subido con éxito.'
            photo_cutted = os.path.join(app.config['UPLOAD_FOLDER'], str(dni), "recortada")
            # photo_redim = os.path.join(app.config['UPLOAD_FOLDER'], str(dni), "redimensionada")
            try:
                mtcnn = getMtcnn()
                device = getDevice()
                resnet = getResnet()
              
                statusFunction, msgF, name_image = cut_image(path_filename, mtcnn, dni) # Recorte, redimesion, luego lo guarda
                if not statusFunction:
                    response_data['message'] = msgF
                    return jsonify(response_data), 400
                response_data['imageUrl'] = f"api/fotos/{dni}/redimensionada/{name_image}"
            #     print(statusFunction, msgF)
                statusFunction1, msgF1 = ProcesarImagenYGuardar(dni, path_filename, mtcnn, device, resnet, path_file=name_image) # Procesa la cara y guarda el embedding
            #     print(statusFunction1, msgF1)
                for code in array_codigos:
                    register_user_photo(
                        "name",
                        "apPaterno",
                        "apMaterno",
                        dni,
                        "dni",
                        "0",
                        "000",
                        str(code)
                        )
            #            data["nombre"],
            #            data["apellidoPaterno"],
            #            data["apellidoMaterno"],
            #            data["dni"],
            #            "dni",
            #            num_schools,
            #            "000",
            #            code
            #           )
                response_data['statusUser'] = '3'
                if statusFunction and statusFunction1:
                    response_data['message'] = 'Imagen procesada correctamente'
                    response_data['statusUser'] = '1'
                    response_data['status'] = 'success'
                elif not statusFunction1:
                    response_data['message'] = msgF1
                elif not statusFunction:
                    response_data['message'] = msgF
                else:
                    response_data['message'] = 'Hubo un error al hacer embedding, recortar imagen, redimensionar imagen'
            except Exception as e:
              print(str(e))
              response_data['statusUser'] = '3'
              return jsonify(response_data), 500
            list_cutted_photos = os.listdir(photo_cutted)
            if len(list_cutted_photos) == 0:
                response_data['message'] = 'El usuario no tiene fotos recortadas'
                response_data['statusUser'] = '2'
                return jsonify(response_data), 400
            # transformation_url = url_for("fotos", filename=f'{dni}/recortada/{list_cutted_photos[0]}')
            # print(transformation_url)
            # url_for_photo = f"fotos/{dni}/recortada/{list_cutted_photos[0]}"
            # response_data['imageUrl'] = url_for_photo
            return jsonify(response_data), 200  # HTTP 200 OK
        return render_template('upload.html')
    
    except Exception as e:
        print(str(e), "ga")
        return jsonify(response_data), 500  # HTTP 500 Internal Server Error


@app.route('/api/validacion_cara', methods=['POST'])
@jwt_required()
def validar_cara():

    '''
    Valida la cara en la foto proporcionada y devuelve el resultado de la validación

    returns:  La respuesta que indica el resultado de la validación de la cara
    '''
        
    # response_data = {'status': 'error', 'message': 'Ocurrió un error durante la carga del archivo.', 'statusUser': '0'}
    response_data = default_response
    try:
        dni_statusFN, dni, dni_code = HandleAnInput(request, "dni", "dni")
        if not dni_statusFN:
            return jsonify(dni), dni_code
        
        photo_statusFN, photo, photo_code = HandleAnInput(request, "image", "file")
        if not photo_statusFN:
            return jsonify(photo), photo_code # Message and HTTP Code
        
        # path_filename = os.path.join(app.config['UPLOAD_FOLDER'], str(dni), "original", photo)
        
        face = Image.open(photo)
        path_dni = os.path.join(app.config['UPLOAD_FOLDER'], str(dni))

        if not os.path.exists(path_dni):
            response_data['message'] = 'No se encuentra el dni'
            response_data['statusUser'] = '3'
            return jsonify(response_data), 400
        photo_embedding = os.path.join(app.config['UPLOAD_FOLDER'], str(dni), "embeddings")
        photo_resized = os.path.join(app.config['UPLOAD_FOLDER'], str(dni), "redimensionada")

        list_embeddings = os.listdir(photo_embedding)
        if len(list_embeddings) == 0:
            response_data['message'] = 'El usuario no tiene embeddings guardados'
            response_data['statusUser'] = '3'
            return jsonify(response_data), 400
        
        list_rezised_photos = os.listdir(photo_resized)
        if len(list_rezised_photos) == 0:
            response_data['message'] = 'El usuario no tiene fotos redimensionadas'
            response_data['statusUser'] = '2'
            return jsonify(response_data), 400
        path_embedding = os.path.join(photo_embedding,list_embeddings[0])
        statusFunction, coseno = id_face(face, path_embedding, getMtcnn(), getDevice(), getResnet()) # Recibe coseno o mensaje de que no hay una cara en la foto
        # print(statusFunction, coseno)
        if statusFunction:
            path_redimensionada = os.path.join(app.config['UPLOAD_FOLDER'], str(dni), "redimensionada")
            list_photos_redimensionadas = os.listdir(path_redimensionada)
            print(list_photos_redimensionadas)
            response_data['message'] = "Comparacion Exitosa"
            response_data['Accuracy'] = coseno
            response_data['urlLastPhoto'] = f"api/fotos/{str(dni)}/redimensionada/{list_photos_redimensionadas[0]}"
            response_data['statusUser'] = "1"
            response_data['status'] = 'success'
        else:
            response_data['message'] = coseno
            response_data['statusUser'] = "3"
            return response_data, 200
        return jsonify(response_data), 200
    except Exception as e:
          return jsonify({
            "message": "Hubo un error al procesar la solicitud.",
            "status": "error",
            "statusUser": "3",
            "error": str(e)
        }), 500

@app.route('/api/buscar_cara', methods=['POST'])
@jwt_required()
def buscar_cara():

    '''
    Busca una cara en la imagen recibida y devuelve un mensaje de estado junto con una lista de similitudes si se encuentra alguna cara

    returns: Un diccionario que contiene el estado de la operacion ('status'), un mensaje ('message') y una lista de similitudes ('lista_similitudes')
    '''
        
    # response_data = {'status': 'error', 'message': 'Ocurrió un error durante la carga del archivo.', 'statusUser': '0'}
    response_data = default_response
    photo_statusFN, photo, photo_code = HandleAnInput(request, "image", "file")
    if not photo_statusFN:
        return jsonify(photo), photo_code # Message and HTTP Code
        
    umbral_selected = request.form.get("umbral")

    # default_umbral = float(0.6)
    # if(umbral_selected != ''):
    #     default_umbral = float(str(umbral_selected))
    # if(umbral_selected is not None ):
    #     default_umbral = float(str(umbral_selected))
    default_umbral = float(umbral_selected) if umbral_selected else float(0.7)
        
    face = Image.open(photo)
    try:
        
        statesU, list_similitudes, msg = search_person(face, getMtcnn(), getDevice(), getResnet(), float(default_umbral))
        response_data['message'] = msg
        response_data['lista_similitudes'] = list_similitudes
        if not statesU:
            return response_data, 400
        return response_data

    except Exception as e:
        print(e)
    return response_data
    
    
@app.route('/api/buscar_recuadro_en_cara', methods=['POST'])
@jwt_required()
def buscar_recuadro_en_cara():

    '''
    Busca un recuadro en una cara en la imagen recibida y devuelve un mensaje de estado junto con las coordenadas del recuadro si se encuentra alguna cara

    returns: Un diccionario que contiene el estado de la operacion ('status'), un mensaje ('message') y las coordenadas del recuadro ('coords')
    '''

    response_data = {'status': 'error', 'message': 'Ocurrió un error durante la carga del archivo.', 'statusUser': '0'}
    file_statusFN, file, file_code = HandleAnInput(request, "image", "file")
    if not file_statusFN:
        return jsonify(file), file_code # Message and HTTP Code
        
    file = request.files['file']
    
    
    image = Image.open(file)
    try:
        statusFunction, msgF, coordsObtained = getSquareOfFace(image, getMtcnn())
        if statusFunction:
            response_data['message'] = msgF
            response_data['coords'] = coordsObtained
            return response_data, 200    
        else:
            response_data['message'] = msgF
            response_data['statusUser'] = 3
            return response_data, 400
    except Exception as e:
        print(str(e))
        return response_data, 400

@app.route('/api/validacion_file', methods=['POST'])
@jwt_required()
def validar_dos_caras():

    '''
    Valida la comparacion de dos caras en las imagenes recibidas y devuelve un mensaje de estado junto con el nivel de precision y la URL del PDF generado si la comparacion es exitosa

    returns: Un diccionario que contiene el estado de la operacion ('status'), un mensaje ('message'), el nivel de precision ('Accuracy' y 'Accuracyeu') y la URL del PDF generado ('url_pdf')
    '''

    # response_data = {'status': 'error', 'message': 'Ocurrió un error durante la carga del archivo.', 'statusUser': '0'}
    response_data = default_response
    try:
        #if request.form.get("dni") is None:
            #response_data['message'] = 'No se proporcionó ningún DNI.'
            #response_data['statusUser'] = '3'
            #return jsonify(response_data), 400
        #dni = request.form.get('dni')
        #if not valid_dni(dni):
            #response_data['message'] = 'Hay un error con el dni'
            #response_data['statusUser'] = '3'
            #return jsonify(response_data), 400
        photo_statusFN, photo, photo_code = HandleAnInput(request, "image", "file")
        if not photo_statusFN:
            return jsonify(photo), photo_code # Message and HTTP Code
        
        photo = request.files['file']
        photo1_statusFN, photo1, photo1_code = HandleAnInput(request, "image", "file1")
        if not photo1_statusFN:
            return jsonify(photo1), photo1_code # Message and HTTP Code
        
        face = Image.open(photo)
        face1 =Image.open(photo1)
        statusFunction, coseno, euclidiana = id_face_compare_two_faces(face,face1, getMtcnn(), getDevice(), getResnet())
        #imgemb=Image.open(imgemb)
        pdf = dat(coseno,euclidiana,face,face1)
        # Recibe coseno o mensaje de que no hay una cara en la foto
        # print(statusFunction, coseno)
        if statusFunction:
            #path_redimensionada = os.path.join(app.config['UPLOAD_FOLDER'], str(dni), "redimensionada")
            #list_photos_redimensionadas = os.listdir(path_redimensionada)
            #print(list_photos_redimensionadas)
            response_data['message'] = "Comparacion Exitosa"
            response_data['Accuracy'] = coseno
            response_data['Accuracyeu'] = euclidiana
            response_data['url_pdf'] = pdf            
            #response_data['urlLastPhoto'] = f"fotos/{str(dni)}/redimensionada/{list_photos_redimensionadas[0]}"
            response_data['statusUser'] = "1"
            response_data['status'] = 'success'
        else:
            response_data['message'] = coseno
            response_data['statusUser'] = "3"
            return response_data, 200
        return jsonify(response_data), 200
    except Exception as e:
          return jsonify({
            "message": "Hubo un error al procesar la solicitud.",
            "status": "error",
            "statusUser": "3",
            "error": str(e)
        }), 500
    

@app.route('/api/comprimirfechas', methods=['GET', 'POST'])
@jwt_required()
def comprimirFechas():
    # response_data = {'status': 'error', 'message': 'Ocurrió un error durante la carga del archivo.', 'statusUser': '0'}
    response_data = default_response
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../fotitos"))
    PARENT_DIR = os.path.dirname(BASE_DIR)
    ZIP_DIR = os.path.join(PARENT_DIR, "all_zip")
    TODAY_DATE = datetime.now().strftime("%Y-%m-%d")
    fp_statusFN, folder_paths, fp_code = HandleAnInput(request, "date", "folder_paths")
    if not fp_statusFN:
        return jsonify(folder_paths), fp_code
    try:
        zip_name = ZipDatesSelected(BASE_DIR,folder_paths.split(','), ZIP_DIR)
        response_data['message'] = "Archivo Comprimidor Exitosamente"
        response_data['status'] = "sucess"
        response_data['urlZip'] = "/api/zipfiles/" + str(zip_name)
    except Exception as e:
        print(str(e))
    return jsonify(response_data), 200

@app.route('/api/comprimir', methods=['GET', 'POST'])
@jwt_required()
def comprimir():

    '''
    Comprime el contenido de las carpetas seleccionadas y guarda el archivo ZIP en una ubicacion especifica

    si se recibe una solicitud POST se espera que contenga la lista de rutas de las carpetas a comprimir
    el contenido de estas carpetas se copia a una carpeta temporal y luego se comprime en un archivo ZIP

    returns: Una respuesta HTTP que renderiza el template 'comprimir.html' con las carpetas disponibles y un mensaje opcional. (GET)
    '''
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../fotitos"))
    PARENT_DIR = os.path.dirname(BASE_DIR)
    ZIP_DIR = os.path.join(PARENT_DIR, "all_zip")
    TODAY_DATE = datetime.now().strftime("%Y-%m-%d")
    ALL_FOLDERS_DIR = os.path.join(ZIP_DIR, f"all_folders-{TODAY_DATE}")
    COPIAS_DIR = os.path.join(ZIP_DIR, f"copias-{TODAY_DATE}")
    
    # Verificar si los directorios existen, y si no, crearlos
    for directory in (ALL_FOLDERS_DIR, COPIAS_DIR):
        if not os.path.exists(directory):
            os.makedirs(directory)

    carpetas = obtener_carpetas(BASE_DIR)
    response_data = {'status': 'error', 'message': 'Ocurrió un error durante la carga del archivo.', 'statusUser': '0'}
    mensaje = None

    if request.method == 'POST':
        fp_statusFN, folder_paths, fp_code = HandleAnInput(request, "date", "folder_paths")
        if not fp_statusFN:
            return jsonify(folder_paths), fp_code
        
        if folder_paths:
            copias = mover_copias([os.path.join(BASE_DIR, folder.strip()) for folder in folder_paths.split(',')], COPIAS_DIR)
            if copias:
                mensaje = f'Se han detectado y movido las siguientes copias a la carpeta "copias": {", ".join(copias)}'
            else:
                mensaje = 'No se encontraron copias para mover.'
            copy_folders_contents([os.path.join(BASE_DIR, folder.strip()) for folder in folder_paths.split(',')], ALL_FOLDERS_DIR)
            pathZip = zip_all_folders(ALL_FOLDERS_DIR, ZIP_DIR)
            # ZipDatesSelected(BASE_DIR,folder_paths.split(','), ZIP_DIR)
            
            mensaje += ' ¡Completado! Se ha comprimido el contenido de las carpetas seleccionadas.'
            response_data['message'] = ' ¡Completado! Se ha comprimido el contenido de las carpetas seleccionadas.'
            response_data['status'] = "success"
            response_data['urlZip'] = "/api/zipfiles/" + str(pathZip)
            return jsonify(response_data), 200

    return render_template('comprimir.html', carpetas=carpetas, mensaje=mensaje)

@app.route('/api/revision', methods=['POST'])
@jwt_required()
def revision():

    '''
    Obtiene la lista de carpetas disponibles para revision

    returns: Una respuesta HTTP que renderiza el template 'comprimir.html' con las carpetas disponibles para revision
    '''
    # response_data = {'status': 'error', 'message': 'Ocurrió un error durante la carga del archivo.', 'statusUser': '0'}
    response_data = default_response
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../fotitos"))
    try:
        carpetas_revision = obtener_carpetas(BASE_DIR)
        response_data['message'] = "Se encontraron carpetas"
        response_data['status'] = "success"
        response_data['folders'] = carpetas_revision
    except Exception as e:
        print(str(e))
        return response_data, 500
    # print(carpetas_revision)
    return jsonify(response_data), 200
    # return render_template('comprimir.html', carpetas_revision=carpetas_revision)

@app.route('/api/getphotos_date', methods=['POST'])
@jwt_required()
def getphotosDate():
    # response_data = {'status': 'error', 'message': 'Ocurrió un error durante la carga del archivo.', 'statusUser': '0'}
    response_data = default_response
    date = request.form.get('date')
    
    if not validar_fecha(str(date)):
        response_data['message'] = "Error en el formato de fecha"
        response_data['statusUser'] = "3"
        return jsonify(response_data), 400
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../fotitos"))
    list_photos = os.listdir(f"{BASE_DIR}/{str(date)}")
    rows_ = []
    for i in range(0, len(list_photos), 1):
        rows_.append({"id": i+1, "dni": list_photos[i], "urlImage": f"api/fotos_V3/{str(date)}/{list_photos[i]}"})
    response_data['message'] = "Se encontraron fotos correctamente"
    response_data['photos'] = rows_
    response_data['statusUser'] = '1' 
    return jsonify(response_data), 200

@app.route('/api/validacion_file1', methods=['POST'])
@jwt_required()
def validar_file1():
    response_data = {'status': 'error', 'message': 'Ocurrió un error durante la carga del archivo.', 'statusUser': '0'}
    try:
        photo_statusFN, photo, photo_code = HandleAnInput(request, "image", "file")
        if not photo_statusFN:
            return jsonify(photo), photo_code
        
        photo1_statusFN, photo1, photo1_code = HandleAnInput(request, "image", "file1")
        if not photo1_statusFN:
            return jsonify(photo1), photo1_code
        
        # path_filename = os.path.join(app.config['UPLOAD_FOLDER'], str(dni), "original", photo)
        face = Image.open(photo)
        face1 =Image.open(photo1)
        width1, height1 = face.size
        width2, height2 = face1.size
        if width1 != width2 or height1 != height2:
            response_data['statusUser'] = '3'
            return jsonify({'error': 'Las dimensiones de las fotos no son iguales'}), 400
        statusFunction, coseno, euclidiana = id_face_compare_two_faces(face,face1, getMtcnn(), getDevice(), getResnet())
        #image_points=image_puntos(face,getMtcnn())
        image_points=points_2d(face,face1)
        #imgemb=Image.open(imgemb)
        pdf = datV2(coseno,euclidiana,face,face1,image_points)
        # Recibe coseno o mensaje de que no hay una cara en la foto
        # print(statusFunction, coseno)
        if statusFunction:
            #path_redimensionada = os.path.join(app.config['UPLOAD_FOLDER'], str(dni), "redimensionada")
            #list_photos_redimensionadas = os.listdir(path_redimensionada)
            #print(list_photos_redimensionadas)
            response_data['message'] = "Comparacion Exitosa"
            response_data['Accuracy'] = coseno
            response_data['Accuracyeu'] = euclidiana                
            #response_data['urlLastPhoto'] = f"fotos/{str(dni)}/redimensionada/{list_photos_redimensionadas[0]}"
            response_data['url_pdf'] = pdf
            response_data['statusUser'] = "1"
            response_data['status'] = 'success'
        else:
            response_data['message'] = coseno
            response_data['statusUser'] = "3"
            return response_data, 200
        return jsonify(response_data), 200
    except Exception as e:
          return jsonify({
            "message": "Hubo un error al procesar la solicitud.",
            "status": "error",
            "statusUser": "3",
            "error": str(e)
        }), 500

@app.route('/api/getFoldersAndQuantity', methods=['POST'])
@jwt_required()
def getFoldersQ():
    # response_data = {'status': 'error', 'message': 'Ocurrió un error durante la carga del archivo.', 'statusUser': '0'}
    response_data = default_response
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../fotitos"))
    list_folders = os.listdir(BASE_DIR)
    if(len(list_folders) == 0):
        response_data['message'] = "No se encontraron carpetas"
        return jsonify(response_data), 500
    rows = []
    for i in range(0, len(list_folders), 1):
        rows.append({"id": i+1, "folder":list_folders[i], "quantity": str(len(os.listdir(os.path.join(BASE_DIR, str(list_folders[i])))))})
    response_data['message'] = "Se encontraron las carpetas"
    response_data['foldersQ'] = rows
    response_data['status'] = "success"
    response_data['statusUser'] = "1"
    return jsonify(response_data), 200 

@app.route('/login', methods=['POST'])
def login():
    username_statusFN, username_data, username_code = HandleAnInput(request, "username", "username")
    if not username_statusFN:
        return jsonify(username_data), username_code
    
    password_statusFN, password_data, password_code = HandleAnInput(request, "password", "password")
    if not password_statusFN:
        return jsonify(password_data), password_code
    
    # username = request.form.get("username")
    # password = request.form.get("password")
    username = username_data
    password = password_data
    # Aquí debes recuperar 'pw_hash' de tu base de datos MySQL usando 'username'
    # pw_hash = "$2y$10$k0RYBE4cXD6r2BWFoT8WDucln9se9p6TfDVUZ1wfyNP36/jxoAdiG"
    pass_in_db = get_user_DB(str(username))[0][1]
    # print(data[0][1])
    try:

        if bcrypt.check_password_hash( pw_hash=pass_in_db, password=password):
            # print(data[0][1])
            access_token = create_access_token(identity=username, fresh=True, expires_delta=timedelta(hours=7))
            refresh_token = create_refresh_token(identity=username)
            default_response['message'] = "Inicio de sesión exitoso"
            default_response['token'] = str(access_token)
            default_response['refresh_token'] = str(refresh_token)
            default_response['data_user'] = {"username": username, "user_rol":"admision", "token": str(access_token), "refresh_token": refresh_token}
            return jsonify(default_response), 200
        else:
            default_response['message'] = "Hay un error en el Usuario o Contraseña"
            return jsonify(default_response), 400
    except Exception as e:
        print(str(e))
        default_response['message'] = "Hubo un error al procesar la imagen"
        return jsonify(default_response), 500

CLAVE_SECRETA_ESPECIFICA = 'lobitos322'
def verificarToken_est(f):
    @wraps(f)
    def decorador(*args, **kwargs):
        token = request.headers.get('eltoken')
        if not token:
            return jsonify({'mensaje': 'No se proporcionó ningún token.'}), 403
        try:
            #print(token)
            jwtf.decode(token, CLAVE_SECRETA_ESPECIFICA, algorithms=['HS256'])
        except Exception as e:
            print(str(e))
            return jsonify({'mensaje': 'Falló la autenticación del token.'}), 500
        return f(*args, **kwargs)
    return decorador

@app.route('/api/validacion_comedor', methods=['POST'])
@verificarToken_est
def validacion_comedor():
    response_data = {'status': 'error', 'message': 'Ocurrió un error durante la carga del archivo.', 'statusUser': '0'}
    try:
        if request.form.get("dni") is None:
            response_data['message'] = 'No se proporcionó ningún DNI.'
            response_data['statusUser'] = '3'
            return jsonify(response_data), 400
        dni = request.form.get('dni')
        if not ValidIdentifiers(dni):
            response_data['message'] = 'Hay un error con el dni'
            response_data['statusUser'] = '1'
            return jsonify(response_data), 400
        if request.form.get("user") is None:
            response_data['message']='No se proporcionó ningún Usuario'
            response_data['statusUser']='3'
            return jsonify(response_data), 400
        user = request.form.get('user')
        # Obtén la carpeta más reciente
        #path = "F:\\prueba"  # Cambiar la ruta al repositorio de fotos
        #latest_folder = get_latest_folder(path)

        # Busca el archivo DNI en la carpeta más reciente
        #path_dni = os.path.join(latest_folder, f        #print(path_dni)
        path='./Faces'
        path_dni = os.path.join(str(path), str(dni))
        if not os.path.exists(path_dni):
            response_data['message'] = 'No se encuentra el dni'
            response_data['statusUser'] = '2'
            return jsonify(response_data)

        path_file = os.path.join(path, str(dni), "redimensionada")
        photo_time = get_latest_f(path_file)
        photo_resized = os.path.join(path_file,photo_time)
        url_photo= watermark(user,photo_resized)
        return jsonify({"data":str(url_photo)})
    except Exception as e:
          return jsonify({
            "message": "Hubo un error al procesar la solicitud.",
            "status": "error",
            "statusUser": "3",
            "error": str(e)
            }), 500
@app.route('/api/validacion_estd', methods=['POST'])
@verificarToken_est
def validacion_estd():
    response_data = {'status': 'error', 'message': 'Ocurrió un error durante la carga del archivo.', 'statusUser': '0'}
    try:
        if request.form.get("dni") is None:
            response_data['message'] = 'No se proporcionó ningún DNI.'
            response_data['statusUser'] = '3'
            return jsonify(response_data), 400
        dni = request.form.get('dni')
        if not ValidIdentifiers(dni):
            response_data['message'] = 'Hay un error con el dni'
            response_data['statusUser'] = '1'
            return jsonify(response_data), 400
        if request.form.get("user") is None:
            response_data['message'] = 'No se proporcionó ningún el Usuario.'
            response_data['statusUser'] = '3'
            return jsonify(response_data), 400
        user = request.form.get('user')
        # Obtén la carpeta más reciente
        #path = "F:\\prueba"  # Cambiar la ruta al repositorio de fotos
        #latest_folder = get_latest_folder(path)

        # Busca el archivo DNI en la carpeta más reciente
        #path_dni = os.path.join(latest_folder, f"{str(dni)}.jpg")
        #print(path_dni)
        path='./Faces'
        path_dni = os.path.join(str(path), str(dni))
        if not os.path.exists(path_dni):
            response_data['message'] = 'No se encuentra el dni'
            response_data['statusUser'] = '2'
            return jsonify(response_data)
        path_file = os.path.join(path, str(dni), "redimensionada")
        photo_time = get_latest_folder(path_file)
        photo_resized = os.path.join(path_file,photo_time)
        url_photo= watermark(user,photo_resized)
        return jsonify({"data":str(url_photo)})
    except Exception as e:
          return jsonify({
            "message": "Hubo un error al procesar la solicitud.",
            "status": "error",
            "statusUser": "3",
            "error": str(e)
        }), 500
@app.route('/developer/test', methods=['POST'])
def dev_test():
# @jwt_required(fresh=False)
    # img_statusFN, img_data, img_code = HandleAnInput(request, "image", "file")
    dni_statusFN, dni_data, dni_code = HandleAnInput(request, "dni", "dni")
    if not dni_statusFN:
        return jsonify(dni_data), dni_code
    codes_statusFN, codes_data, codes_code = HandleAnInput(request, "codes_school", "codigos")
    if not codes_statusFN:
        # default_response["message"] = codes_data
        return jsonify(codes_data), codes_code
    array_codigos = codes_data.split(',')
    # if not img_statusFN:
    #     return jsonify(img_data), img_code
    # return jsonify(img_data.filename)
    # date_statusFN, date_data, date_code = HandleAnInput(request, "date", "folder_paths")
    # if not date_statusFN:
        # return jsonify(date_data), date_code
    # return date_data
    status_D, data, num_schools = HandleDataUser(getInfoFromAPI(dni_data))
    if not status_D:
        return jsonify(data), 400


    for code in array_codigos:
        register_user_photo(
            data["nombre"],
            data["apellidoPaterno"],
            data["apellidoMaterno"],
            data["dni"],
            "dni",
            num_schools,
            "000",
            code
            )
    # current_user = get_jwt_identity()
    # img = Image.open("./all_faces/12345697.jpg")
    # image = putWatermark(img, "SONIA DEL CARMEN MILAGROS NIÑO DE GUZMAN MOLINA")
    # urlBase64 = to_data_url(image)
    # print(data["data"],len(data["data"]))
    # image.save("pruebaFinal.jpg")
    # return urlBase64
    # return current_user
    #print(data)
    return data
