from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get('/')
def home():
    message = """
    Bienvenido.<br>
    <span style='color: black;'>Añade </span>
    <span style='color: red; font-weight: bold;'>/docs</span>
    <span style='color: black;'> a la url actual para ir a la documentación 
    de la API y probar los endpoints</span>
    """
    return HTMLResponse(content=message)