import firebase_admin
from firebase_admin import credentials, storage, firestore
import uuid
import datetime

# uploads image and associated data to firebase
# requires a database key to be located in the same folder
# data required: classifications, contributer email, latitude/longitude
def uploadImage( image, data_dict ):

    # add valid  firebase credentials
    creds = credentials.Certificate('crafttest-7acfe-firebase-adminsdk-sm233-f98a48f12d.json')
    firebase_admin.initialize_app(creds, { 'storageBucket': 'crafttest-7acfe.appspot.com' })
    database = firestore.client()

    # upload image with random name/retrieve image URL
    bucket = storage.bucket( )
    random_name = str(uuid.uuid1())
    blob = bucket.blob( 'images/' + random_name )
    blob.upload_from_filename( image )

    # upload associated metadata
    data_dict['imageUrl'] = blob.public_url
    database.collection("classifications").document( random_name ).set( data_dict )

# function test
classification_dict = {
    'allClassifications': 
        { 'Black Mesa': 0.022120080888271332,
        'Dogoszhi': 0.0007386765792034566,
        'Flagstaff' : 0.003873235546052456,
        'Kana\'a' : 0.968483030796051,
        'Kayenta' : 0.0004921318031847477,
        'Sosi' : 0.00003520447717164643,
        'Tusayan': 0.004257689695805311},
    'imageUrl': "",
    'latitude': 35.18479092220293,
    'longitude': -111.66314619318408,
    'primaryClassification': "Kayenta",
    'timestamp': datetime.datetime.now(),
    'userId': "6mInHOhGmdTaGdiOLb4bKoCsu3e2"
}

uploadImage("./photos/image.png", classification_dict)
