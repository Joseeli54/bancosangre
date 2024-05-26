import { Component, OnInit } from '@angular/core';
import { HttpClient , HttpHeaders} from  '@angular/common/http';
import { SweetAlertOptions } from 'sweetalert2';
import { UserServices } from '../services/userservices.services';
import { ActivatedRoute, ActivatedRouteSnapshot, RouterStateSnapshot, Router } from '@angular/router';
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

  public clickInsertarImagen(){
    $('#avatar').click();
  }

  public register() {
    $('#register_button').html("<li class='fa fa-spinner fa-spin fa-1x'> </li>");

    let data = {
        "name" :  this.name,
        "apellido" :  this.apellido,
        "username" :  this.username,
        "email" :  this.email,
        "password" :  this.password,
        "role" :  this.role,
        "avatar" :  '',
        "avatar_public_id" :  '',
        "telefono" :  this.telefono,
        "type" :  this.type,
        "direccion" : this.direccion
    };

    this.userService
      .postUrl('register', data)
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
