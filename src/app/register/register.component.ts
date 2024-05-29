import { Component, OnInit } from '@angular/core';
import { HttpClient , HttpHeaders} from  '@angular/common/http';
import { SweetAlertOptions } from 'sweetalert2';
import { UserServices } from '../services/userservices.services';
import { ActivatedRoute, ActivatedRouteSnapshot, RouterStateSnapshot, Router } from '@angular/router';
import { environment } from '../../environments/environment';
import Swal from 'sweetalert2';
import $ from 'jquery';

@Component({
  selector: 'app-register',
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css']
})
export class RegisterComponent {

  name: string = '';
  apellido: string = '';
  username: string = '';
  email: string = '';
  dateBirthday: string = '';
  password: string = '';
  confirmPassword: string = '';
  role: string = '';
  type: string = '';
  avatar: string = '';
  direccion: string = '';
  telefono: string = '';
  delete_token: string = "";
  public_id: string = "";
  public archivo : any = [];
  private apiUrl : string = environment.apiUrl;

  public dia : any = "";
  public meses : any = [
    {"code" : "01", "mes" : "Enero"}, 
    {"code" : "02", "mes" : "Febrero"}, 
    {"code" : "03", "mes" : "Marzo"}, 
    {"code" : "04", "mes" :"Abril"}, 
    {"code" : "05", "mes" :"Mayo"}, 
    {"code" : "06", "mes" :"Junio"}, 
    {"code" : "07", "mes" :"Julio"}, 
    {"code" : "08", "mes" :"Agosto"}, 
    {"code" : "09", "mes" :"Septiembre"}, 
    {"code" : "10", "mes" :"Octubre"}, 
    {"code" : "11", "mes" :"Noviembre"}, 
    {"code" : "12", "mes" :"Diciembre"}, 
  ];
  public mes : any = "01";
  public ano : any = "";

  urlPreview: any = '';
  public file : any = [];
  
  passwordError: boolean = false;
  bSend = false;

  constructor(public userService: UserServices, 
    private router: Router, 
    private  httpClient:  HttpClient) {}

  ngOnInit() {
    this.role = "USER";
    this.type = "Paciente";
  }

  EqualsPassword(){
    if(this.password == this.confirmPassword && 
      ((this.password !== '' || this.confirmPassword !== '') && (this.password !== undefined || this.confirmPassword !== undefined))){
      return true;
    }

    return 0;
  }

  InequalsPassword(){
    if(this.password !== this.confirmPassword && 
      ((this.password !== '' || this.confirmPassword !== '') && (this.password !== undefined || this.confirmPassword !== undefined))){
      return true;
    }

    return 0;
  }

  public register() {
    $('#register_button').html("<li class='fa fa-spinner fa-spin fa-1x'> </li>");

    let dateBirth = this.dia.toString() + "/" + this.mes + "/" + this.ano.toString(); 
    var data = new FormData();

    data.append("name", this.name);
    data.append("apellido", this.apellido);
    data.append("username", this.username);
    data.append("email", this.email);
    data.append("password", this.password);
    data.append("role", this.role);
    data.append("dateBirth", dateBirth);
    data.append("avatar", this.archivo);
    data.append("avatar_public_id", '');
    data.append("telefono", this.telefono);
    data.append("type", this.type);
    data.append("direccion", this.direccion);
    data.append("apiUrl", this.apiUrl);

    this.userService
      .postUrlFiles('register', data)
      .then(response => {
            $('#register_button').html("Registrarse");
            console.log(response);

            this.bSend = true;
            this.messageSuccessfully();
            this.router.navigate(['/login']);
      })
      .catch(data =>{
            $('#register_button').html("Registrarse");
            this.errorOcurred(data.error.err.message)
      });
  }

  private messageSuccessfully() {
    let config: SweetAlertOptions = {
      title: 'Usuario registrado satisfactoriamente',
      icon: 'success',
      showConfirmButton: false,
      timer: 2500
    };

    Swal.fire(config).then(result => {
    });
  }

  public capturarImagen(event : any){
    var  fileName : any;
    const archivoEncontrado = event.target.files[0];
    this.archivo = archivoEncontrado;

    fileName = this.archivo['name'];

    if (event.target.files && event.target.files[0]) {
      var reader = new FileReader();

      reader.onload = (event: ProgressEvent) => {
        this.urlPreview = (<FileReader>event.target).result;
        //this.compressFile(this.urlPreview, fileName)
      }

      reader.readAsDataURL(event.target.files[0]);
    }
  }

  public clickEliminarImagen(){

    var inputAvatar = <HTMLInputElement> document.getElementById('avatar');

    if(inputAvatar !== null){
       inputAvatar.value = '';
    }

    this.archivo = [];
    this.urlPreview = '';
  }


  // public compressFile(image : any, fileName : any) {
  //       var orientation = -1;

  //       console.warn('Size in bytes is now:',  this.imageCompress.byteCount(image)/(1024*1024));
        
  //       this.imageCompress.compressFile(image, orientation, 50, 50).then(
  //       result => {
  //           this.urlPreview = result;
            
  //           console.warn('Size in bytes after compression:',  this.imageCompress.byteCount(result)/(1024*1024));
  //           // create file from byte
  //           const imageName = fileName;
            
  //           // call method that creates a blob from dataUri
  //           // const imageBlob = this.dataURItoBlob(this.urlPreview.split(',')[1]);

  //           //imageFile created below is the new compressed file which can be send to API in form data
  //           this.archivo = new File([result], imageName, { type: 'image/jpeg' });
  //       });

  // }

  public clickInsertarImagen(){
    $('#avatar').click();
  }
  
  private errorOcurred(message : any) {
    let config: SweetAlertOptions = {
      title: message,
      icon: 'error',
      showConfirmButton: true
    };

    Swal.fire(config).then(result => {
    });
  }

  public backClicked() {
    window.history.back();
  }

}
