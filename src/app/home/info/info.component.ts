import { Component } from '@angular/core';

@Component({
  selector: 'app-info',
  templateUrl: './info.component.html',
  styleUrls: ['./info.component.css']
})
export class InfoComponent {
  public bLogger = false;

  constructor() { }

  ngOnInit() {

      var rol = localStorage.getItem('role');
      var email = localStorage.getItem('email');
      
      if (!this.isNullOrUndefined(rol) && !this.isNullOrUndefined(email)) {
          this.bLogger = true;
      }
  }

  public isNullOrUndefined(item : any){
    if(item == null || item == "" || item == 0 || item.length == 0 || item.length == undefined){
      return true;
    }

    return false;
  }
}
