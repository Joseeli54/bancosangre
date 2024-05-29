import { Component, OnInit, ElementRef, AfterViewInit, Renderer2} from '@angular/core';
import { FormControl, FormGroup, FormBuilder, Validators } from '@angular/forms';
import { NgbModal, ModalDismissReasons } from '@ng-bootstrap/ng-bootstrap';
import { UserServices } from '../../services/userservices.services';
import { Router, ActivatedRoute } from '@angular/router';
import { SweetAlertOptions } from 'sweetalert2';
import Swal from 'sweetalert2';
import $ from 'jquery';

@Component({
  selector: 'app-profile',
  templateUrl: './profile.component.html',
  styleUrls: ['./profile.component.css']
})
export class ProfileComponent {

  public propertyForm : FormGroup | undefined;
  public username : any; //Del perfil del usuario
  public roleCurrent : any;
  public usernameCurrent : any; //Del usuario en linea
  public usuario : any;
  public file : any = [];
  public urlPreview: any = '';


  // Variables para edicion
  public id : any = "";
  public names: any = "";
  public apellidos: any = "";
  public correo: any = "";
  public telefono: any = "";
  public cargo: any = "";
  public avatar: any = "";
  public avatar_public_id: any = "";
  public usernameOld : any = "";
  public role : any = "";
  public products : any[] = [];
  public orders : any[] = [];
  public modalReference: any;
  public closeResult: string = "";

  constructor(private modalService: NgbModal, private _formBuilder: FormBuilder, 
    private router: ActivatedRoute, private Redirect: Router,
    public userService: UserServices) { }

  ngOnInit() {
    localStorage.setItem('view', '4');
    this.getUser();
  }  

  public open(content : any) {
    this.modalReference = this.modalService.open(content, { size: 'lg', centered: true });

    this.modalReference.result.then((result : any) => {
      this.closeResult = `Closed with: ${result}`;
    }, (reason : any) => {
      this.closeResult = `Dismissed ${this.getDismissReason(reason)}`;
    });
  }

  private getDismissReason(reason: any): string {
    if (reason === ModalDismissReasons.ESC) {
      return 'by pressing ESC';
    } else if (reason === ModalDismissReasons.BACKDROP_CLICK) {
      return 'by clicking on a backdrop';
    } else {
      return  `with: ${reason}`;
    }
  }

  public getUser(){

    var username = localStorage.getItem('username') || "";

    this.userService.getUrl('users/username/{username}', [username])
    .then(data => {
      this.id= data._id;
      this.names= data.name;
      this.apellidos = data.apellido;
      this.correo= data.email;
      this.telefono= data.telefono;
      this.cargo= data.puesto;
      this.avatar= data.avatar;

      if(this.avatar == null || this.avatar == ""){
        this.avatar = "/assets/img/perfil-user.jpg";
      }

      this.avatar_public_id= "";
      this.usernameCurrent = data.username;
      this.username = data.username;
      this.role= data.role;

      this.usuario = data;
    })
    .catch(err => {
      console.log(err);
    })
  }

  public createFormChangePassword(){
      this.propertyForm = this._formBuilder.group({
        password: ['', Validators.required],
        passwordNew: ['', Validators.required]
      });
  }

  public changeUser(){

    let data = {
      "name": this.names,
      "apellido": this.apellidos,
      "email": this.correo,
      "telefono": this.telefono
    };

    this.userService.putUrl('users/username/{username}', data, [this.username])
    .then(response => {
      console.log(response)
      if(response._id !== undefined){

          this.names = response.name;
          this.apellidos = response.apellido;
          this.correo = response.email;
          this.telefono = response.telefono;
          
          this.modalReference.close();
          this.getUser();

          if(document.location.href.indexOf('myprofile') !== -1){
              var URL = '/home/myprofile/'+localStorage.getItem('username');
          }else{
              var URL = '/home/profile/'+localStorage.getItem('username');
          }

          this.editingSuccessfully();
          
          this.Redirect.navigateByUrl(URL);
      }
    })
    .catch(data =>{
        this.errorOcurred(data.error);
    });

  }


  private messageSwal(title : any) {
    let config: SweetAlertOptions = {
      title: title,
      icon: 'success',
      showConfirmButton: false,
      timer: 2500
    };

    Swal.fire(config).then(result => {
    });
  }

  private messageSuccessfully() {
    let config: SweetAlertOptions = {
      title: 'Contraseña cambiada satisfactoriamente',
      icon: 'success',
      showConfirmButton: false,
      timer: 2500
    };

    Swal.fire(config).then(result => {
    });
  }

  private editingSuccessfully() {
    let config: SweetAlertOptions = {
      title: 'Los datos del usuario han sido editados satisfactoriamente',
      icon: 'success',
      showConfirmButton: false,
      timer: 2500
    };

    Swal.fire(config).then(result => {
    });
  }

  private deleteSuccessfully() {
    let config: SweetAlertOptions = {
      title: 'El usuario ha sido eliminado satisfactoriamente',
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

}
